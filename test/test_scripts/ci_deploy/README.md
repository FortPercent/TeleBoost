# 准备
## 初始化 ci 仓库（因为每个人路径不一样，所以直接clone，不用submodule）
git clone ssh://{研发云用户名}@code.srdcloud.cn:29418/P24HQASYF0004/AI-Infra/helm-define-ci

## 安装telego (本机+云桌面)
### 本机安装

git clone ssh://{研发云用户名}@code.srdcloud.cn:29418/P24HQASYF0004/AI-Infra/telego_release

执行 telego_release 中对应系统的安装脚本

- win: （管理员终端）

  ```
  set-executionpolicy remotesigned 
  powershell ./install.ps1
  ```

- linux:
  
  ```
  bash install.sh 
  
  或 
  
  python3 install.py
  ```
### 云桌面安装

- win powershell: 
  
  ```
  $env:MAIN_NODE_IP = "10.127.16.5"; $response = Invoke-WebRequest -Uri "http://$($env:MAIN_NODE_IP):8003/bin_telego/install.ps1" -UseBasicParsing; $script = [System.Text.Encoding]::UTF8.GetString($response.Content); Invoke-Expression $script
  ```

- linux bash
  
  ```
  export MAIN_NODE_IP=10.127.16.5
  curl -s http://${MAIN_NODE_IP}:8003/bin_telego/install.sh | bash
  ```

在安装完成后启动一次telego配置项目路径

## 链接 telego project
```
windows:
cmd /c mklink /D {telego project directory}/k8s_teletron_ci {ci_deploy目录绝对路径}

linux:
ln -s ci_deploy 目录到 {telego project directory}/k8s_teletron_ci
```

# 更新helm配置（本机）
```
telego cmd --cmd deploy/k8s_teletron_ci/prepare
```

# 上传ci依赖镜像，对于一个集群环境只需进行一次
```
telego cmd --cmd deploy/k8s_teletron_ci/upload
```

# 部署helm (云桌面）
```
telego cmd --cmd deploy/k8s_teletron_ci/apply/{集群名（见kubeconfig或telego ui）}
```

# 使用kubepi或vscode查看对应pod执行状态