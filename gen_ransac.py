# Standard library imports
import abc
import numpy as np

class Model(abc.ABC):
    def __init__(self) -> None:
        super().__init__()
        self.N = 1

    @abc.abstractmethod
    def build(self, points) -> None:
        pass
    
    @abc.abstractmethod
    def calc_errors(self, points):
        pass

import cv2
class EllipseModel(Model):
    
    def __init__(self) -> None:
        super().__init__()
        self.N = 5
        self.ellipse = ((0,0),(0,0),0)

    def build(self, points) -> None:
        self.ellipse= cv2.fitEllipse(points)
        return self.ellipse
    
    def calc_errors(self, points):
        #   Utils function from https://github.com/jmtyszka/mrgaze/blob/master/mrgaze/fitellipse.py
        def _Geometric2Conic(ellipse):
            """
            Geometric to conic parameter conversion

            References
            ----
            Adapted from Swirski's ConicSection.h
            https://bitbucket.org/Leszek/pupil-tracker/
            """

            # Ellipse tuple has form ( ( x0, y0), (bb, aa), phi_b_deg) )
            # Where aa and bb are the major and minor axes, and phi_b_deg
            # is the CW x to minor axis rotation in degrees
            (x0,y0), (bb, aa), phi_b_deg = ellipse

            # Semimajor and semiminor axes
            a, b = aa/2, bb/2

            # Convert phi_b from deg to rad
            phi_b_rad = phi_b_deg * np.pi / 180.0

            # Major axis unit vector
            ax, ay = -np.sin(phi_b_rad), np.cos(phi_b_rad)

            # Useful intermediates
            a2 = a*a
            b2 = b*b

            #
            # Conic parameters
            #
            if a2 > 0 and b2 > 0:

                A = ax*ax / a2 + ay*ay / b2;
                B = 2*ax*ay / a2 - 2*ax*ay / b2;
                C = ay*ay / a2 + ax*ax / b2;
                D = (-2*ax*ay*y0 - 2*ax*ax*x0) / a2 + (2*ax*ay*y0 - 2*ay*ay*x0) / b2;
                E = (-2*ax*ay*x0 - 2*ay*ay*y0) / a2 + (2*ax*ay*x0 - 2*ax*ax*y0) / b2;
                F = (2*ax*ay*x0*y0 + ax*ax*x0*x0 + ay*ay*y0*y0) / a2 + (-2*ax*ay*x0*y0 + ay*ay*x0*x0 + ax*ax*y0*y0) / b2 - 1;

            else:

                # Tiny dummy circle - response to a2 or b2 == 0 overflow warnings
                A,B,C,D,E,F = (1,0,1,0,0,-1e-6)

            # Compose conic parameter array
            conic = np.array((A,B,C,D,E,F))

            return conic
        def _ConicFunctions(pnts, ellipse):
            """
            Calculate various conic quadratic curve support functions

            General 2D quadratic curve (biquadratic)
            Q = Ax^2 + Bxy + Cy^2 + Dx + Ey + F
            For point on ellipse, Q = 0, with appropriate coefficients

            Parameters
            ----
            pnts : n x 2 array of floats
            ellipse : tuple of tuples

            Returns
            ----
            distance : array of floats
            grad : array of floats
            absgrad : array of floats
            normgrad : array of floats

            References
            ----
            Adapted from Swirski's ConicSection.h
            https://bitbucket.org/Leszek/pupil-tracker/
            """

            # Suppress invalid values
            np.seterr(invalid='ignore')

            # Convert from geometric to conic ellipse parameters
            conic = _Geometric2Conic(ellipse)

            # Row vector of conic parameters (Axx, Axy, Ayy, Ax, Ay, A1) (1 x 6)
            C = np.array(conic)

            # Extract vectors of x and y values
            x, y = pnts[:,0], pnts[:,1]

            # Construct polynomial array (6 x n)
            X = np.array( ( x*x, x*y, y*y, x, y, np.ones_like(x) ) )

            # Calculate Q/distance for all points (1 x n)
            distance = C.dot(X)

            # Quadratic curve gradient at (x,y)
            # Analytical grad of Q = Ax^2 + Bxy + Cy^2 + Dx + Ey + F
            # (dQ/dx, dQ/dy) = (2Ax + By + D, Bx + 2Cy + E)

            # Construct conic gradient coefficients vector (2 x 3)
            Cg = np.array( ( (2*C[0], C[1], C[3]), (C[1], 2*C[2], C[4]) ) )

            # Construct polynomial array (3 x n)
            Xg = np.array( (x, y, np.ones_like(x) ) )

            # Gradient array (2 x n)
            grad = Cg.dot(Xg)

            # Normalize gradient -> unit gradient vector
            # absgrad = np.apply_along_axis(np.linalg.norm, 0, grad)
            absgrad = np.sqrt(np.sqrt(grad[0,:]**2 + grad[1,:]**2))
            normgrad = grad / absgrad

            return distance, grad, absgrad, normgrad        
        def _EllipseError(pnts, ellipse):
            """
            Ellipse fit error function
            """

            # Suppress divide-by-zero warnings
            np.seterr(divide='ignore')

            # Calculate algebraic distances and gradients of all points from fitted ellipse
            distance, grad, absgrad, normgrad = _ConicFunctions(pnts, ellipse)

            # Calculate error from distance and gradient
            # See Swirski et al 2012
            # TODO : May have to use distance / |grad|^0.45 - see Swirski source

            # Gradient array has x and y components in rows (see ConicFunctions)
            err = distance / absgrad

            return err
        def _EllipseNormError(pnts, ellipse):
            """
            Error normalization factor, alpha

            Normalizes cost to 1.0 at point 1 pixel out from minor vertex along minor axis
            """

            # Ellipse tuple has form ( ( x0, y0), (bb, aa), phi_b_deg) )
            # Where aa and bb are the major and minor axes, and phi_b_deg
            # is the CW x to minor axis rotation in degrees
            (x0,y0), (bb,aa), phi_b_deg = ellipse

            # Semiminor axis
            b = bb/2

            # Convert phi_b from deg to rad
            phi_b_rad = phi_b_deg * np.pi / 180.0

            # Minor axis vector
            bx, by = np.cos(phi_b_rad), np.sin(phi_b_rad)

            # Point one pixel out from ellipse on minor axis
            p1 = np.array( (x0 + (b + 1) * bx, y0 + (b + 1) * by) ).reshape(1,2)

            # Error at this point
            err_p1 = _EllipseError(p1, ellipse)

            # Errors at provided points
            err_pnts = _EllipseError(pnts, ellipse)

            return err_pnts / err_p1
        return _EllipseNormError(points, self.ellipse)

class FixedCenterCircleModel(Model):
    
    def __init__(self, center) -> None:
        super().__init__()
        self.N = 1
        self.center = np.array(center)
        self.radius = 0

    def build(self, points) -> None:
        self.radius= np.linalg.norm(points[0]-self.center)
        return self.radius
    
    def calc_errors(self, points):
        radii = np.linalg.norm(points-self.center, axis=-1)
        return np.abs(radii-self.radius)
    
def ransac_fit(model: Model, pnts, success_probabilities=0.98, outliers_ratio = 0.45, inliers_thres = 3.0):
    # Debug flag
    DEBUG = True

    np.seterr(invalid='ignore')

    max_norm_err_sq = inliers_thres * inliers_thres

    best_model_pts_ids = None
    best_n_inliers = 0
    best_inliers_error = np.inf

    n_pnts = len(pnts)
      
    # Break if too few points to fit model
    if n_pnts < model.N:
        return None

    # compute maximum iterations from parameters
    max_itts = round(np.log(1-success_probabilities) / np.log(1-np.power(1-outliers_ratio, 5)))
    # if(DEBUG):
    #     print("Computed max iters:", max_itts)

    # Ransac iterations
    for iter in range(max_itts):
        
        # Select N points at random
        pnts_id = np.random.randint(0,n_pnts,model.N)
        sample_pnts = pnts[pnts_id]

        # Build model from selected points
        model.build(sample_pnts)

        # Calculate normalized errors for all points
        errors = model.calc_errors(pnts)

        # Identify inliers
        inliers = np.nonzero(errors**2 < max_norm_err_sq)[0]


        n_inliers = inliers.size

        # Protect fitting from too few points
        if n_inliers < model.N:
            continue

        # Update best model
        if(n_inliers> best_n_inliers):
            best_n_inliers = n_inliers
            best_model_pts_ids = pnts_id
            best_inliers_error = np.mean(errors[inliers])
        elif(n_inliers == best_n_inliers):
            mean_err = np.mean(errors[inliers])
            if(mean_err < best_inliers_error):
                best_inliers_error = mean_err
                best_model_pts_ids = pnts_id

    return model.build(pnts[best_model_pts_ids])
