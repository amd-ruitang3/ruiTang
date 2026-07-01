# 查询谁 / 哪个容器在占用 GPU

## 给 AI agent 的指引

用户问“谁在占 GPU / who is using the machine”时，**自动执行**下面流程，把结果（用户名/容器名）
直接报给用户，不要只贴命令；命令报错就如实说卡在哪一步，不要猜。

**关键：命令里的 PID 一律用 `rocm-smi` 拿到的真实数字填好再给用户，绝不留 `<PID>` 占位符。**

## 流程

1. 拿 PID：`rocm-smi --showpids`（PID 为宿主机视角，含每个进程 VRAM 占用）。
   - VRAM USED 单位是**字节**，报给用户时换算成 GB（÷10^9，如 `4560494592` → ~4.56 GB）。
   - 用户只关心占用最高的，就挑 VRAM USED 最大的 PID 往下查。
   - **默认只查显存占用最高的那一个 PID**，不要把所有 PID 都列出来反查，除非用户明确要求查全部。
   - 反查提示要 **Docker 和 Podman 两套都给全**（外加「通用」判断段），不要替用户在 Docker/Podman 之间二选一——当前容器内探测不到宿主机装的是哪个。
   - 多进程常属于**同一容器/用户**（如 8 卡 TP 推理会拆成一堆进程）。反查前先看 cgroup / 命令行是否同源，别把一个人的活动报成好几拨人。
2. 判断当前 shell 能否反查：
   - 能（有 docker/podman，`/proc/<PID>` 可读）→ 直接执行下面「反查命令」，把结果报给用户。
   - **不能**（docker/podman 都没有、`/proc/<PID>` 读不到）→ 多半在容器内，PID 属于宿主机命名空间。
     **不要瞎猜**，把 PID + VRAM 报给用户，说明当前 shell 反查不了，并把「反查命令」给用户去宿主机跑。

## 反查命令（按 Docker / Podman 分类，PID 填真实值，下例用 172965）

**通用（先跑，判断属于谁 + 命令行）：**
```bash
cat /proc/172965/cgroup                 # cgroup 含容器 id 或 user-<UID>.slice
ps -o user=,cmd= -p 172965              # 进程属主 + 命令行
```

**Docker：**
```bash
docker ps -q | xargs -I{} sh -c 'docker top {} | grep -q 172965 && echo {}'   # 得到 CID
docker inspect --format '{{.Name}}' <CID>                                     # CID → 容器名
```

**Podman（AAC / rootless，先查用户名，再查容器）：**
```bash
# 1) 先查是谁（别人的进程到此通常够用）：
cat /proc/172965/cgroup                 # 找 user-<UID>.slice
id -nu <UID>                            # UID → 用户名
# 2) 想进一步定位容器：
pstree -s -a -p 172965 | grep -oE '\-c [0-9a-f]{6,}'   # 挖出容器 id
podman ps -a | grep <容器id前6位>                        # 反查容器名
```

> 只有 `docker ps` 那条的 `{}` 是 xargs 占位符（容器 ID，自动填）；PID 必须自己填真实值。

## 一键脚本（宿主机上跑，自动遍历所有 PID）

```bash
#!/usr/bin/env bash
set -uo pipefail
pids=$(rocm-smi --showpids | grep -oE '^[0-9]+' | sort -u)
[ -z "$pids" ] && { echo "无进程占用 GPU"; exit 0; }
for pid in $pids; do
  printf "PID %s -> " "$pid"
  if command -v docker >/dev/null 2>&1; then
    cid=$(docker ps -q | xargs -I{} sh -c "docker top {} 2>/dev/null | grep -q '\\b$pid\\b' && echo {}" 2>/dev/null | head -n1)
    [ -n "$cid" ] && { echo "Docker 容器 $(docker inspect --format '{{.Name}}' "$cid" 2>/dev/null | sed 's#^/##') ($cid)"; continue; }
  fi
  if command -v podman >/dev/null 2>&1; then
    cid=$(pstree -s -a -p "$pid" 2>/dev/null | grep -oE '\-c [0-9a-f]{6,}' | head -n1 | awk '{print $2}')
    if [ -n "$cid" ]; then
      m=$(podman ps -a --format '{{.ID}} {{.Names}}' 2>/dev/null | grep "^${cid:0:6}")
      [ -n "$m" ] && { echo "Podman 容器 $m"; continue; }
    fi
  fi
  uid=$(grep -oE 'user-[0-9]+\.slice' "/proc/$pid/cgroup" 2>/dev/null | grep -oE '[0-9]+' | head -n1)
  if [ -n "$uid" ]; then echo "用户 $(id -nu "$uid" 2>/dev/null || echo "UID $uid")"
  else echo "属主 $(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')（未匹配容器）"; fi
done
```

> 脚本未在本环境实测；Docker 分支需 daemon 权限，`/proc`、`pstree` 需在宿主机执行。

## 备注

- AAC 环境用 Podman；判断顺序：先 Docker，查不到再 Podman。
- 容器内 shell 常见结果：docker/podman 都没有、`/proc/<PID>` 不可读 —— 直接走「不能反查」分支。
