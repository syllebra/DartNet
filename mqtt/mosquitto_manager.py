#!/usr/bin/env python3
"""
MQTT Broker with mDNS Advertisement - Windows Enhanced Version
Fixes common Windows dependency issues and provides better diagnostics
Compatible with Windows, macOS, and Linux
"""

import subprocess
import signal
import sys
import time
import socket
import tempfile
import os
import argparse
import threading
import platform
import shutil
import urllib.request

try:
    from zeroconf import IPVersion, ServiceInfo, Zeroconf
except ImportError:
    print("Error: zeroconf library not found. Install with: pip install zeroconf")
    sys.exit(1)

# Windows-specific imports
if platform.system() == "Windows":
    import winreg

    try:
        import psutil

        HAS_PSUTIL = True
    except ImportError:
        print("Info: psutil not found. Install with: pip install psutil (recommended for Windows)")
        HAS_PSUTIL = False


class WindowsMosquittoInstaller:
    """Helper class to handle Mosquitto installation on Windows"""

    def __init__(self, debug=False):
        self.debug = debug
        self.temp_dir = os.path.join(os.environ.get("TEMP", "C:\\temp"), "mqtt_broker_installer")

    def debug_print(self, message):
        if self.debug:
            print(f"INSTALLER DEBUG: {message}")

    def check_dependencies(self):
        """Check if required dependencies are installed"""
        issues = []

        # Check Visual C++ Redistributable
        if not self._check_vcredist():
            issues.append("Visual C++ Redistributable 2015-2022 (x64) is missing")

        # Check OpenSSL
        if not self._check_openssl():
            issues.append("OpenSSL libraries are missing or incompatible")

        return issues

    def _check_vcredist(self):
        """Check if Visual C++ Redistributable is installed"""
        try:
            # Check registry for VC++ redistributable
            key_paths = [
                r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
                r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
                r"SOFTWARE\Classes\Installer\Dependencies\Microsoft.VS.VC_RuntimeMinimumVSU_amd64,v14",
            ]

            for key_path in key_paths:
                try:
                    winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                    return True
                except FileNotFoundError:
                    continue
            return False
        except Exception as e:
            self.debug_print(f"VC++ check failed: {e}")
            return False

    def _check_openssl(self):
        """Check if OpenSSL is available"""
        try:
            # Common OpenSSL DLL locations
            openssl_paths = [
                "C:\\Windows\\System32\\libssl-1_1-x64.dll",
                "C:\\Windows\\System32\\libcrypto-1_1-x64.dll",
                "C:\\Program Files\\OpenSSL-Win64\\bin\\libssl-1_1-x64.dll",
                "C:\\Program Files\\OpenSSL-Win64\\bin\\libcrypto-1_1-x64.dll",
            ]

            for path in openssl_paths[:2]:  # Check system32 first
                if os.path.exists(path):
                    return True
            return False
        except Exception as e:
            self.debug_print(f"OpenSSL check failed: {e}")
            return False

    def install_portable_mosquitto(self):
        """Download and setup a portable version of Mosquitto with dependencies"""
        try:
            print("📦 Downloading portable Mosquitto with dependencies...")
            os.makedirs(self.temp_dir, exist_ok=True)

            # Download mosquitto portable version
            mosquitto_url = "https://mosquitto.org/files/binary/win64/mosquitto-2.0.18-install-windows-x64.exe"
            installer_path = os.path.join(self.temp_dir, "mosquitto-installer.exe")

            print("⬇️  Downloading Mosquitto installer...")
            urllib.request.urlretrieve(mosquitto_url, installer_path)

            # Extract using 7zip if available, otherwise try manual extraction
            mosquitto_dir = os.path.join(self.temp_dir, "mosquitto")

            print("📂 Extracting Mosquitto...")
            # Try to extract the installer (it's actually a zip-like archive)
            try:
                # Use Windows built-in expand command
                result = subprocess.run(
                    ["expand", installer_path, "-F:*", mosquitto_dir], capture_output=True, text=True
                )

                if result.returncode != 0:
                    raise Exception("Expand failed")

            except Exception:
                # Fallback: run installer silently
                try:
                    subprocess.run([installer_path, "/S", f"/D={mosquitto_dir}"], check=True, timeout=60)
                except Exception as e:
                    raise Exception(f"Failed to install Mosquitto: {e}")

            # Find mosquitto.exe in the extracted/installed files
            mosquitto_exe = None
            for root, dirs, files in os.walk(mosquitto_dir):
                if "mosquitto.exe" in files:
                    mosquitto_exe = os.path.join(root, "mosquitto.exe")
                    break

            if not mosquitto_exe or not os.path.exists(mosquitto_exe):
                raise Exception("Mosquitto executable not found after installation")

            print(f"✅ Portable Mosquitto installed at: {mosquitto_exe}")
            return mosquitto_exe

        except Exception as e:
            print(f"❌ Failed to install portable Mosquitto: {e}")
            return None

    def suggest_manual_fixes(self):
        """Provide manual fix suggestions"""
        print("\n🔧 MANUAL FIX SUGGESTIONS:")
        print("=" * 50)

        print("\n1. INSTALL VISUAL C++ REDISTRIBUTABLE:")
        print("   Download and install from:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")

        print("\n2. INSTALL OPENSSL:")
        print("   Download from: https://slproweb.com/products/Win32OpenSSL.html")
        print("   Choose: Win64 OpenSSL v3.x.x Light")

        print("\n3. ALTERNATIVE: USE DOCKER:")
        print("   docker run -d -p 1883:1883 --name mosquitto eclipse-mosquitto")

        print("\n4. ALTERNATIVE: USE WSL2:")
        print("   wsl --install")
        print("   wsl")
        print("   sudo apt update && sudo apt install mosquitto")

        print("\n5. ALTERNATIVE: USE CHOCOLATEY:")
        print("   choco install mosquitto")


class MQTTBrokerManager:
    def __init__(self, port=1883, service_name="MQTT Broker", websocket_port=None, debug=False, custom_config=None):
        self.port = port
        self.websocket_port = websocket_port
        self.service_name = service_name
        self.mosquitto_process = None
        self.zeroconf = None
        self.service_info = None
        self.config_file = None
        self.custom_config = custom_config
        self.platform = platform.system()
        self.mosquitto_path = None
        self.debug = debug
        self.installer = WindowsMosquittoInstaller(debug) if self.platform == "Windows" else None

    def debug_print(self, message):
        """Print debug messages if debug mode is enabled"""
        if self.debug:
            print(f"DEBUG: {message}")

    def check_port_availability(self, port):
        """Check if a port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return True
        except OSError:
            return False

    def find_mosquitto_executable(self):
        """Find Mosquitto executable on different platforms with enhanced Windows support"""
        if self.mosquitto_path:
            return self.mosquitto_path

        # Try common executable names
        executable_names = ["mosquitto", "mosquitto.exe"]

        # Check if mosquitto is in PATH
        for name in executable_names:
            path = shutil.which(name)
            if path:
                self.mosquitto_path = path
                self.debug_print(f"Found mosquitto in PATH: {path}")
                # Test if it works
                if self.test_mosquitto_executable(path):
                    return path
                else:
                    print(f"⚠️  Found mosquitto at {path} but it has dependency issues")

        # Platform-specific search paths
        if self.platform == "Windows":
            return self._find_mosquitto_windows()
        elif self.platform == "Darwin":  # macOS
            return self._find_mosquitto_macos()
        else:  # Linux and other Unix-like systems
            return self._find_mosquitto_linux()

    def _find_mosquitto_windows(self):
        """Enhanced Windows Mosquitto finding with automatic fixes"""
        # Common Windows installation paths
        common_paths = [
            r"C:\Program Files\mosquitto\mosquitto.exe",
            r"C:\Program Files (x86)\mosquitto\mosquitto.exe",
            r"C:\mosquitto\mosquitto.exe",
            r"C:\tools\mosquitto\mosquitto.exe",
            os.path.join(os.environ.get("TEMP", "C:\\temp"), "mqtt_broker_installer", "mosquitto", "mosquitto.exe"),
        ]

        # Test existing installations
        for path in common_paths:
            if os.path.exists(path):
                self.debug_print(f"Testing mosquitto at: {path}")
                if self.test_mosquitto_executable(path):
                    self.mosquitto_path = path
                    return path
                else:
                    print(f"⚠️  Found mosquitto at {path} but it has dependency issues")

        # Check dependencies and try to fix
        print("🔍 Checking Windows dependencies...")
        if self.installer:
            issues = self.installer.check_dependencies()
            if issues:
                print("❌ Dependency issues found:")
                for issue in issues:
                    print(f"   - {issue}")

                # Try to install portable version
                print("\n🚀 Attempting to download and setup portable Mosquitto...")
                portable_path = self.installer.install_portable_mosquitto()
                if portable_path and self.test_mosquitto_executable(portable_path):
                    self.mosquitto_path = portable_path
                    return portable_path

                # If automatic fix failed, show manual instructions
                self.installer.suggest_manual_fixes()

        return None

    def _find_mosquitto_macos(self):
        """Find Mosquitto on macOS systems"""
        common_paths = [
            "/usr/local/bin/mosquitto",
            "/opt/homebrew/bin/mosquitto",
            "/opt/local/bin/mosquitto",
            "/usr/bin/mosquitto",
        ]

        for path in common_paths:
            if os.path.exists(path):
                self.mosquitto_path = path
                return path

        return None

    def _find_mosquitto_linux(self):
        """Find Mosquitto on Linux systems"""
        common_paths = [
            "/usr/bin/mosquitto",
            "/usr/local/bin/mosquitto",
            "/usr/sbin/mosquitto",
            "/usr/local/sbin/mosquitto",
        ]

        for path in common_paths:
            if os.path.exists(path):
                self.mosquitto_path = path
                return path

        return None

    def test_mosquitto_executable(self, mosquitto_exe):
        """Test if mosquitto executable works with enhanced Windows error detection"""
        try:
            # Set up environment for Windows
            env = os.environ.copy()
            if self.platform == "Windows":
                # Add the mosquitto directory to PATH for DLL loading
                mosquitto_dir = os.path.dirname(mosquitto_exe)
                env["PATH"] = mosquitto_dir + os.pathsep + env.get("PATH", "")

            result = subprocess.run([mosquitto_exe, "--help"], capture_output=True, text=True, timeout=10, env=env)

            self.debug_print(f"Mosquitto help exit code: {result.returncode}")
            if result.stdout and self.debug:
                self.debug_print(f"Mosquitto help stdout: {result.stdout[:200]}")
            if result.stderr and self.debug:
                self.debug_print(f"Mosquitto help stderr: {result.stderr[:200]}")

            # Windows-specific error code handling
            if self.platform == "Windows":
                if result.returncode == 3221226505:  # 0xC0000409 - missing DLL
                    print("❌ DLL dependency error detected")
                    return False
                elif result.returncode == 3221225781:  # 0xC0000135 - DLL not found
                    print("❌ Required DLL not found")
                    return False
                elif result.returncode == 193:  # Invalid Win32 application
                    print("❌ Architecture mismatch (32-bit vs 64-bit)")
                    return False

            # Success conditions
            if result.returncode == 0 or "mosquitto" in result.stdout.lower():
                return True
            else:
                self.debug_print(f"Mosquitto help failed with code {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Mosquitto executable test timed out")
            return False
        except Exception as e:
            self.debug_print(f"Failed to test mosquitto executable: {e}")
            return False

    def create_mosquitto_config(self):
        """Create a temporary Mosquitto configuration file with corrected syntax"""
        # Create platform-appropriate temp directory
        if self.platform == "Windows":
            temp_dir = os.path.join(os.environ.get("TEMP", "C:\\temp"), "mosquitto")
        else:
            temp_dir = "/tmp/mosquitto"

        try:
            os.makedirs(temp_dir, exist_ok=True)
            if self.platform == "Windows":
                os.chmod(temp_dir, 0o777)
        except Exception as e:
            print(f"Warning: Could not create temp directory {temp_dir}: {e}")

        # CORRECTED configuration with proper Mosquitto syntax
        config_content = f"""# =================================================================
# MQTT Broker Configuration - Fixed Version
# =================================================================

# General configuration
allow_anonymous true
allow_zero_length_clientid true
auto_id_prefix auto-

# =================================================================
# Listeners
# =================================================================
listener {self.port} 0.0.0.0
protocol mqtt

# =================================================================
# Security and Connection Settings
# =================================================================
max_connections -1
max_inflight_messages 20
max_queued_messages 100
max_packet_size 268435456

# =================================================================
# Persistence (disabled for simplicity)
# =================================================================
persistence false

# =================================================================
# Logging
# =================================================================
# log_dest stdout
# log_type error
# log_type warning
# log_timestamp true
# connection_messages false
"""

        # Add WebSocket listener if specified
        if self.websocket_port:
            config_content += f"""
# =================================================================
# WebSocket Listener
# =================================================================
listener {self.websocket_port}
protocol websockets
"""

        # Create temporary config file
        self.config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False)
        self.config_file.write(config_content)
        self.config_file.close()

        try:
            os.chmod(self.config_file.name, 0o644)
        except Exception:
            pass

        self.debug_print(f"Created config file: {self.config_file.name}")
        if self.debug:
            print("📋 Configuration content:")
            print(config_content)

        return self.config_file.name

    def start_mosquitto(self):
        """Start the Mosquitto MQTT broker with enhanced Windows support"""
        mosquitto_exe = self.find_mosquitto_executable()
        if not mosquitto_exe:
            if self.platform == "Windows":
                print("\n❌ Could not find or fix Mosquitto installation.")
                print("📖 Please follow the manual installation steps above.")
            raise Exception(self._get_installation_instructions())

        # Test the executable first
        if not self.test_mosquitto_executable(mosquitto_exe):
            raise Exception(f"Mosquitto executable at {mosquitto_exe} is not working properly")

        # Check if port is available
        if not self.check_port_availability(self.port):
            raise Exception(
                f"Port {self.port} is already in use. "
                f"Please choose a different port or stop the service using that port."
            )

        # Use custom config if provided, otherwise create a new one
        if self.custom_config:
            if not os.path.exists(self.custom_config):
                raise Exception(f"Custom configuration file not found: {self.custom_config}")
            config_path = self.custom_config
            print(f"📄 Using custom config file: {config_path}")
        else:
            config_path = self.create_mosquitto_config()

        try:
            cmd = [mosquitto_exe, "-c", config_path]
            if self.debug:
                cmd.append("-v")
            print(f"🚀 Starting Mosquitto MQTT broker on port {self.port}...")
            print(f"📍 Using executable: {mosquitto_exe}")
            print(f"⚙️  Config file: {config_path}")

            if self.debug:
                print(f"🔧 Command: {' '.join(cmd)}")

            # Enhanced Windows process creation
            if self.platform == "Windows":
                # Set up environment with proper DLL path
                env = os.environ.copy()
                mosquitto_dir = os.path.dirname(mosquitto_exe)
                env["PATH"] = mosquitto_dir + os.pathsep + env.get("PATH", "")

                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

                self.mosquitto_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Combine streams for better output
                    universal_newlines=True,
                    startupinfo=startupinfo,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    cwd=mosquitto_dir,
                    env=env,
                )
            else:
                self.mosquitto_process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
                )

            # Give mosquitto time to start
            time.sleep(2)

            # Check if process is still running
            if self.mosquitto_process.poll() is not None:
                try:
                    stdout, _ = self.mosquitto_process.communicate(timeout=3)
                    error_output = stdout if stdout else "No error output available"

                    # Enhanced error analysis
                    if "permission denied" in error_output.lower():
                        error_output += "\n\n💡 TIP: Try running as Administrator"
                    elif "bind" in error_output.lower():
                        error_output += (
                            f"\n\n💡 TIP: Port {self.port} is in use. Try: netstat -an | findstr {self.port}"
                        )
                    elif "config" in error_output.lower() or "error" in error_output.lower():
                        error_output += "\n\n💡 TIP: Configuration file issue detected"
                        if self.debug:
                            error_output += f"\n📋 Config file location: {config_path}"

                except subprocess.TimeoutExpired:
                    error_output = "Process terminated without output (timeout)"

                raise Exception(
                    f"❌ Mosquitto failed to start.\n\nError details:\n{error_output}\n\n"
                    f"Return code: {self.mosquitto_process.returncode}"
                )

            print("✅ Mosquitto MQTT broker started successfully!")

            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_mosquitto_output)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

            # Test connection
            if self._test_broker_connection():
                print("✅ Broker connection test successful")
            else:
                print("⚠️  Warning: Broker may not be accepting connections")

        except Exception as e:
            self.cleanup()
            raise e

    def _test_broker_connection(self):
        """Test if the broker is accepting connections"""
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(3)
            result = test_socket.connect_ex(("localhost", self.port))
            test_socket.close()
            return result == 0
        except Exception:
            return False

    def _get_installation_instructions(self):
        """Get platform-specific installation instructions"""
        if self.platform == "Windows":
            return (
                "❌ MOSQUITTO INSTALLATION REQUIRED\n\n"
                "📥 AUTOMATIC SOLUTIONS:\n"
                "1. Run this script with --auto-install flag (if available)\n"
                "2. Use Docker: docker run -d -p 1883:1883 eclipse-mosquitto\n"
                "3. Use Chocolatey: choco install mosquitto\n\n"
                "🔧 MANUAL INSTALLATION:\n"
                "1. Download from: https://mosquitto.org/download/\n"
                "2. Install Visual C++ Redistributable 2015-2022 (x64)\n"
                "3. Install OpenSSL for Windows\n"
                "4. Verify with: mosquitto --help"
            )
        elif self.platform == "Darwin":
            return "Install mosquitto: brew install mosquitto"
        else:
            return (
                "Install mosquitto broker:\n"
                "Ubuntu/Debian: sudo apt install mosquitto\n"
                "CentOS/RHEL: sudo yum install mosquitto\n"
                "Arch Linux: sudo pacman -S mosquitto"
            )

    def _monitor_mosquitto_output(self):
        """Monitor mosquitto output in a separate thread"""
        if self.mosquitto_process:
            try:
                while self.mosquitto_process.poll() is None:
                    line = self.mosquitto_process.stdout.readline()
                    if line:
                        line = line.strip()
                        if self.debug or "error" in line.lower() or "warning" in line.lower():
                            print(f"📋 Mosquitto: {line}")
                    time.sleep(0.1)
            except Exception as e:
                self.debug_print(f"Error monitoring mosquitto output: {e}")

    def get_local_ip(self):
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def start_mdns_advertisement(self):
        """Start mDNS advertisement for the MQTT broker"""
        try:
            self.zeroconf = Zeroconf(ip_version=IPVersion.V4Only)

            local_ip = self.get_local_ip()
            hostname = socket.gethostname()

            service_type = "_mqtt._tcp.local."
            service_name = f"{self.service_name.replace(' ', '-')}._mqtt._tcp.local."

            properties = {
                "name": self.service_name.encode("utf-8"),
                "version": "1.0".encode("utf-8"),
                "protocol": "mqtt".encode("utf-8"),
            }

            if self.websocket_port:
                properties["websocket_port"] = str(self.websocket_port).encode("utf-8")

            self.service_info = ServiceInfo(
                service_type,
                service_name,
                addresses=[socket.inet_aton(local_ip)],
                port=self.port,
                properties=properties,
                server=f"{hostname}.local.",
            )

            self.zeroconf.register_service(self.service_info)

            print("✅ mDNS advertisement started")
            print(f"   🏷️  Service: {service_name}")
            print(f"   🌐 Address: {local_ip}:{self.port}")

            if self.websocket_port:
                print(f"   🔌 WebSocket: {local_ip}:{self.websocket_port}")

        except Exception as e:
            print(f"❌ Failed to start mDNS advertisement: {e}")

    def stop(self):
        """Stop the MQTT broker and mDNS advertisement"""
        print("\n🛑 Shutting down...")

        # Stop mDNS advertisement
        if self.zeroconf and self.service_info:
            try:
                self.zeroconf.unregister_service(self.service_info)
                self.zeroconf.close()
                print("✅ mDNS advertisement stopped")
            except Exception as e:
                print(f"❌ Error stopping mDNS: {e}")

        # Stop Mosquitto
        if self.mosquitto_process:
            try:
                if self.platform == "Windows" and HAS_PSUTIL:
                    try:
                        process = psutil.Process(self.mosquitto_process.pid)
                        process.terminate()
                        process.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                else:
                    self.mosquitto_process.terminate()
                    try:
                        self.mosquitto_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if self.platform == "Windows":
                            subprocess.run(
                                ["taskkill", "/F", "/PID", str(self.mosquitto_process.pid)], capture_output=True
                            )
                        else:
                            self.mosquitto_process.kill()

                print("✅ Mosquitto broker stopped")
            except Exception as e:
                print(f"❌ Error stopping Mosquitto: {e}")

        self.cleanup()

    def cleanup(self):
        """Clean up temporary files"""
        if self.config_file and os.path.exists(self.config_file.name):
            try:
                os.unlink(self.config_file.name)
                self.debug_print(f"Cleaned up config file: {self.config_file.name}")
            except Exception as e:
                self.debug_print(f"Failed to clean up config file: {e}")

    def run(self):
        """Run the MQTT broker with mDNS advertisement"""
        try:
            self.start_mosquitto()
            self.start_mdns_advertisement()

            print("\n🎉 MQTT Broker is running!")
            print(f"📡 MQTT Port: {self.port}")
            if self.websocket_port:
                print(f"🌐 WebSocket Port: {self.websocket_port}")
            print(f"🏷️  mDNS Name: {self.service_name}")
            print(f"🔗 Connect using: mqtt://{self.get_local_ip()}:{self.port}")
            print("\n⏹️  Press Ctrl+C to stop the broker")

            while True:
                if self.mosquitto_process and self.mosquitto_process.poll() is not None:
                    print("❌ Mosquitto process died unexpectedly")
                    try:
                        stdout, _ = self.mosquitto_process.communicate(timeout=1)
                        if stdout:
                            print(f"Final output: {stdout}")
                    except Exception:
                        pass
                    break
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n👋 Received interrupt signal")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.stop()


def signal_handler(sig, frame, broker_manager):
    """Handle interrupt signals"""
    broker_manager.stop()
    sys.exit(0)


def setup_signal_handlers(broker_manager):
    """Setup signal handlers for different platforms"""
    if platform.system() == "Windows":
        signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, broker_manager))
        try:
            signal.signal(signal.SIGBREAK, lambda sig, frame: signal_handler(sig, frame, broker_manager))
        except AttributeError:
            pass
    else:
        signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, broker_manager))
        signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, broker_manager))


def main():
    parser = argparse.ArgumentParser(description="Enhanced MQTT broker with mDNS advertisement and Windows fixes")
    parser.add_argument("-p", "--port", type=int, default=1883, help="MQTT broker port (default: 1883)")
    parser.add_argument("-w", "--websocket-port", type=int, help="WebSocket port for MQTT over WebSockets")
    parser.add_argument("-n", "--name", type=str, default="MQTT Broker", help="Service name for mDNS advertisement")
    parser.add_argument("--mosquitto-path", type=str, help="Full path to mosquitto executable")
    parser.add_argument("-c", "--config", type=str, help="Custom Mosquitto configuration file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    broker_manager = MQTTBrokerManager(
        port=args.port,
        websocket_port=args.websocket_port,
        service_name=args.name,
        debug=args.debug,
        custom_config=args.config,
    )

    if args.mosquitto_path:
        broker_manager.mosquitto_path = args.mosquitto_path

    setup_signal_handlers(broker_manager)

    print(f"🖥️  Platform: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")

    if platform.system() == "Windows":
        print("🔧 Windows enhanced mode with dependency checking")

    broker_manager.run()


if __name__ == "__main__":
    main()
