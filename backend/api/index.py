from fastapi import FastAPI
app = FastAPI()

@app.get("/api/v1/health")
def health():
    return {"status": "ok", "message": "Vercel routing works!"}

@app.get("/api/test")
def test():
    return {"status": "ok"}
