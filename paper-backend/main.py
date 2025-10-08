from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 初始化 FastAPI 应用
app = FastAPI()

# 配置允许跨域访问的前端域名
# React 开发环境通常运行在 http://localhost:5000
origins = [
    "http://localhost:5000",  # 必须包含协议（http/https）和端口
    # 生产环境添加实际域名，例如："https://your-react-app.com"
]

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许指定的前端域名访问
    allow_credentials=True,  # 允许前端携带 cookies（如认证信息）
    allow_methods=["*"],     # 允许所有 HTTP 方法（GET/POST/PUT/DELETE 等）
    allow_headers=["*"],     # 允许所有请求头（如 Authorization、Content-Type 等）
)

# 示例接口（供 React 前端调用）
@app.get("/api/hello")
async def hello():
    return {"message": "Hello from FastAPI (允许跨域访问)"}

@app.post("/api/login")
async def login(username: str, password: str):
    # 实际项目中这里会验证用户名密码
    if username == "test" and password == "123":
        return {"code": 200, "token": "react-fastapi-token", "message": "登录成功"}
    return {"code": 400, "message": "账号或密码错误"}
