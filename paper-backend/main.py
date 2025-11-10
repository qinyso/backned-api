from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from datetime import datetime, timedelta, timezone
import jwt
from passlib.context import CryptContext
from typing import Optional, List, Dict
import subprocess
import os
import json
import logging
from pathlib import Path

# ------------------------------
# 初始化配置（新增文件上传配置）
# ------------------------------
app = FastAPI()

# 配置日志（新增文件上传相关日志记录）
logging.basicConfig(
    filename="inference.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------
# 原有配置项（保留不变）
# ------------------------------
# JWT配置
SECRET_KEY = "5f8a8f8d8b6e4a2c9d8f7e6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
# 密码哈希配置
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
MAX_PASSWORD_LENGTH = 72
# 跨域配置
origins = [
    "http://localhost:5000",
    "http://localhost:5173",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# 新增：文件上传配置（替换原Flask配置）
# ------------------------------
UPLOAD_FOLDER = './upload_files'  # 文件保存目录
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB 上传限制
# 创建上传目录（不存在则自动创建）
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# ------------------------------
# 算法推理相关配置（保留不变）
# ------------------------------
TRAINER = "BiomedCoOp_BiomedCLIP"
SCRIPT_PATH = "/home/ooze/BiomedCoOp/train_demo.py"
CONFIG_FILE = "/home/ooze/BiomedCoOp/configs/trainers/biomedcoop.yaml"
DEFAULT_DATASET_CONFIG = "/home/ooze/BiomedCoOp/configs/datasets/btmri.yaml"
DEFAULT_ROOT = "/home/ooze/BiomedCoOp/data"
MAX_TIMEOUT = 300
ALLOWED_MODEL_ROOT = "/home/ooze/BiomedCoOp/output"

# ------------------------------
# 数据模型定义（新增文件上传响应模型）
# ------------------------------
class UserCreate(BaseModel):
    username: str
    password: str
    email: str
    full_name: Optional[str] = None

class UserUpdate(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None

class UserInDB(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    hashed_password: str
    created_at: datetime
    model_config = ConfigDict(mutable=True)

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    created_at: datetime

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    code: int = 200
    token: str
    token_type: str = "bearer"
    message: str = "登录成功"
    user_info: UserResponse

class InferenceRequest(BaseModel):
    model_dir: str
    load_epoch: int
    dataset_config: Optional[str] = DEFAULT_DATASET_CONFIG
    root: Optional[str] = DEFAULT_ROOT

class InferenceResponse(BaseModel):
    code: int = 200
    msg: str = "推理成功"
    data: dict

# 新增：文件上传响应模型
class UploadResponse(BaseModel):
    code: int = 200
    msg: str = "文件上传成功"
    data: Dict[str, str]  # 存储文件名和保存路径

# ------------------------------
# 内存存储（保留不变）
# ------------------------------
fake_db = {
    "users": [],
    "next_id": 1
}

# ------------------------------
# 工具函数（保留不变，新增文件名安全处理函数）
# ------------------------------
def get_password_hash(password: str) -> str:
    if len(password) > MAX_PASSWORD_LENGTH:
        password = password[:MAX_PASSWORD_LENGTH]
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    if len(plain_password) > MAX_PASSWORD_LENGTH:
        plain_password = plain_password[:MAX_PASSWORD_LENGTH]
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user_by_username(username: str) -> Optional[UserInDB]:
    return next((u for u in fake_db["users"] if u.username == username), None)

def get_user_by_id(user_id: int) -> Optional[UserInDB]:
    return next((u for u in fake_db["users"] if u.id == user_id), None)

# 新增：安全处理文件名（避免路径穿越攻击，类似Flask的secure_filename）
def secure_filename(filename: str) -> str:
    """过滤危险字符，生成安全的文件名"""
    # 移除路径分隔符
    filename = filename.replace("/", "").replace("\\", "").replace("..", "")
    # 处理空文件名
    if not filename:
        return f"unknown_file_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return filename

def validate_inference_params(params: InferenceRequest) -> bool:
    abs_model_dir = os.path.abspath(params.model_dir)
    abs_dataset_config = os.path.abspath(params.dataset_config)
    abs_root = os.path.abspath(params.root)
    
    if not abs_model_dir.startswith(ALLOWED_MODEL_ROOT):
        return False
    if not os.path.isdir(abs_model_dir):
        return False
    if not os.path.isfile(abs_dataset_config):
        return False
    if not os.path.isdir(abs_root):
        return False
    if params.load_epoch <= 0:
        return False
    checkpoint_file = os.path.join(abs_model_dir, f"epoch_{params.load_epoch}.pth")
    if not os.path.isfile(checkpoint_file):
        return False
    return True

def parse_inference_output(output: str) -> dict:
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"raw_output": output.strip()}

# ------------------------------
# 认证依赖（保留不变）
# ------------------------------
async def get_current_user(token: str = Depends(lambda x: x.headers.get("Authorization", "").replace("Bearer ", ""))):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证信息或未登录",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = get_user_by_username(username)
    if user is None:
        raise credentials_exception
    return user

# ------------------------------
# 原有接口（保留不变）
# ------------------------------
@app.post("/api/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    if get_user_by_username(user.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已被占用"
        )
    
    new_user = UserInDB(
        id=fake_db["next_id"],
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=get_password_hash(user.password),
        created_at=datetime.now(timezone.utc)
    )
    
    fake_db["users"].append(new_user)
    fake_db["next_id"] += 1
    return new_user

@app.post("/api/login", response_model=TokenResponse)
async def login(login_data: LoginRequest):
    user = get_user_by_username(login_data.username)
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名或密码错误"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    return {
        "token": access_token,
        "user_info": user
    }

@app.get("/api/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserInDB = Depends(get_current_user)):
    return current_user

@app.get("/api/users", response_model=List[UserResponse])
async def get_all_users(current_user: UserInDB = Depends(get_current_user)):
    return fake_db["users"]

@app.get("/api/users/{user_id}", response_model=UserResponse)
async def get_user_detail(user_id: int, current_user: UserInDB = Depends(get_current_user)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    if user.id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限访问该用户信息"
        )
    return user

@app.put("/api/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    update_data: UserUpdate,
    current_user: UserInDB = Depends(get_current_user)
):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    if user.id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限更新该用户信息"
        )
    
    if update_data.email:
        user.email = update_data.email
    if update_data.full_name is not None:
        user.full_name = update_data.full_name
    if update_data.password:
        user.hashed_password = get_password_hash(update_data.password)
    
    return user

@app.delete("/api/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    current_user: UserInDB = Depends(get_current_user)
):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    if user.id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限删除该用户"
        )
    
    fake_db["users"] = [u for u in fake_db["users"] if u.id != user_id]

@app.get("/api/hello")
async def hello():
    return {"message": "Hello from FastAPI (支持用户管理+算法推理+文件上传)"}

@app.post("/api/inference", response_model=InferenceResponse)
async def run_inference(
    request: InferenceRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    logger.info(f"用户 {current_user.username} 发起推理请求：{request.dict()}")
    
    if not validate_inference_params(request):
        error_msg = "参数无效（可能是路径不存在、检查点缺失或非法路径）"
        logger.error(f"用户 {current_user.username} 推理失败：{error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    
    cmd = [
        "python", SCRIPT_PATH,
        "--trainer", TRAINER,
        "--config-file", CONFIG_FILE,
        "--dataset-config-file", request.dataset_config,
        "--root", request.root,
        "--eval-only",
        "--model-dir", request.model_dir,
        "--load-epoch", str(request.load_epoch)
    ]
    logger.info(f"执行推理命令：{' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=MAX_TIMEOUT
        )
        
        inference_data = parse_inference_output(result.stdout)
        logger.info(f"用户 {current_user.username} 推理成功")
        
        return {
            "code": 200,
            "msg": "推理成功",
            "data": inference_data
        }
    
    except subprocess.TimeoutExpired:
        error_msg = f"推理超时（超过{MAX_TIMEOUT}秒）"
        logger.error(f"用户 {current_user.username} 推理失败：{error_msg}")
        raise HTTPException(status_code=504, detail=error_msg)
    
    except subprocess.CalledProcessError as e:
        error_msg = f"脚本执行失败：{e.stderr.strip()}"
        logger.error(f"用户 {current_user.username} 推理失败：{error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    except Exception as e:
        error_msg = f"接口异常：{str(e)}"
        logger.error(f"用户 {current_user.username} 推理失败：{error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

# ------------------------------
# 新增：文件上传接口（带登录认证）
# ------------------------------
@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),  # 接收单个文件
    current_user: UserInDB = Depends(get_current_user)  # 强制登录
):
    """
    文件上传接口（需登录后调用）
    - 请求头需携带 Token: Bearer <登录返回的token>
    - 支持单个文件上传，最大100MB
    """
    # 1. 记录上传日志
    logger.info(f"用户 {current_user.username} 发起文件上传：{file.filename}")
    
    # 2. 校验文件大小
    file_size = 0
    try:
        # 读取文件内容并计算大小
        contents = await file.read()
        file_size = len(contents)
        if file_size > MAX_UPLOAD_SIZE:
            error_msg = f"文件过大（最大支持100MB），当前文件大小：{file_size / 1024 / 1024:.2f}MB"
            logger.error(f"用户 {current_user.username} 上传失败：{error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
    finally:
        # 重置文件指针（避免后续读取异常）
        await file.seek(0)
    
    # 3. 安全处理文件名
    safe_filename = secure_filename(file.filename)
    
    # 4. 构建保存路径
    save_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    
    # 5. 保存文件
    try:
        with open(save_path, "wb") as f:
            f.write(contents)
        logger.info(f"用户 {current_user.username} 上传成功，文件保存至：{save_path}")
    except Exception as e:
        error_msg = f"文件保存失败：{str(e)}"
        logger.error(f"用户 {current_user.username} 上传失败：{error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    # 6. 返回结果
    return {
        "code": 200,
        "msg": "文件上传成功",
        "data": {
            "filename": safe_filename,
            "save_path": save_path,
            "file_size": f"{file_size / 1024:.2f}KB"
        }
    }

# ------------------------------
# 新增：支持多文件上传接口（可选）
# ------------------------------
@app.post("/api/upload-multiple", response_model=UploadResponse)
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    current_user: UserInDB = Depends(get_current_user)
):
    """支持多文件同时上传"""
    upload_results = []
    for file in files:
        try:
            contents = await file.read()
            file_size = len(contents)
            if file_size > MAX_UPLOAD_SIZE:
                upload_results.append(f"{file.filename}：文件过大，跳过上传")
                continue
            
            safe_filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, safe_filename)
            
            with open(save_path, "wb") as f:
                f.write(contents)
            upload_results.append(f"{safe_filename}：上传成功")
            await file.seek(0)
        except Exception as e:
            upload_results.append(f"{file.filename}：上传失败 - {str(e)}")
    
    logger.info(f"用户 {current_user.username} 批量上传结果：{upload_results}")
    return {
        "code": 200,
        "msg": "批量上传完成",
        "data": {"results": upload_results}
    }

if __name__ == "__main__":
    import uvicorn
    # 启动服务，端口保持8000（原FastAPI端口）
    uvicorn.run(app, host="0.0.0.0", port=8000)