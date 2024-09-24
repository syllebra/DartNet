# https://github.com/docker/docker-py
#pip install docker==2.1.0

# https://medium.com/@mariovanrooij/adding-https-to-fastapi-ad5e0f9e084e

from fastapi import FastAPI, Body, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer

import time

app = FastAPI()

next_button_visible = False

# Test if API is Live
@app.get("/")
async def root():
    return {"message": "Alive"}
  
# Run a background test docker and retruns docker id
@app.get("/hit")
async def hit(cb:str|None = None, cmd: str|None = None):
    global next_button_visible
    print("Dummy received: cb:", cb, " cmd:",cmd)
    ret = {}
    if(cb is not None):
        ret["cb"] = cb
        if(cb == "click"):
            next_button_visible = not next_button_visible
        ret["button_state"] = next_button_visible
           
        
    if(cmd  is not None):
        ret["cmd"] = cmd

    #print(json.dumps(ret.attrs, indent=4))
    time.sleep(0.3)
    return ret


if __name__ == "__main__":
    import uvicorn

    # Start server on port 8080
    config = uvicorn.Config("dummy_game:app", port=8088, host="0.0.0.0", log_level="info")#,
    #              ssl_keyfile="./test_key.pem", 
    #              ssl_certfile="./test_cert.pem")
    server = uvicorn.Server(config)
    server.run()