# คู่มือย้ายงานไปเครื่อง RTX 4060 (สำหรับเจ้าของงาน) — 2026-09-19

## สิ่งที่ต้องขนไป
| อะไร | อยู่ที่ไหน | หมายเหตุ |
|---|---|---|
| `Deep_CNN_handoff_2026-09-19.tar.gz` (+ `.sha256`, `.manifest.txt`) | `~/` บนโน้ตบุ๊ก | ~1 GB: โค้ด + `.git` + dataset (ไม่รวม `__MACOSX`) + `data/` (cache/synth/external/splits) + checkpoint ที่จำเป็น 8 ไฟล์ |
| โค้ดล่าสุด | GitHub private **https://github.com/Eakkachad/thaichar72** | branch `main`; tarball มี `.git` ที่ชี้ `origin` ไปที่นี่แล้ว → `git pull` จะได้ของใหม่ |

ตรวจไฟล์หลังคัดลอก (วางไฟล์ .tar.gz กับ .sha256 ไว้โฟลเดอร์เดียวกัน): `sha256sum -c Deep_CNN_handoff_2026-09-19.tar.gz.sha256`
บน Windows: `certutil -hashfile Deep_CNN_handoff_2026-09-19.tar.gz SHA256` แล้วเทียบกับค่าในไฟล์ .sha256

## ติดตั้งบนเครื่อง 4060 (แนะนำ WSL2 Ubuntu 22.04/24.04)
1. Windows: อัปเดต **NVIDIA driver ≥ 560** (รองรับ CUDA 12.6) — เช็คด้วย `nvidia-smi` ใน PowerShell
2. เปิด WSL2: PowerShell (admin) → `wsl --install -d Ubuntu` → เข้า Ubuntu แล้ว `nvidia-smi` ต้องเห็นการ์ด (ไม่ต้องลง CUDA toolkit ใน WSL — wheel ของ torch มี runtime มาเอง)
3. ใน Ubuntu:
   ```bash
   sudo apt update && sudo apt install -y git pigz
   curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
   mkdir -p ~/work && cd ~/work
   tar -I pigz -xf /mnt/c/Users/<คุณ>/Downloads/Deep_CNN_handoff_2026-09-19.tar.gz   # หรือ tar xzf
   cd Deep_CNN && gh auth login && git pull      # ดึงโค้ดล่าสุดจาก GitHub (repo private: pull/push ต้องมี token)
   uv sync                                       # เลือก torch cu126 อัตโนมัติบน WSL/Windows (ดาวน์โหลด ≈ 4 GB บน WSL2, ≈ 3 GB บน Windows; ต้องมีที่ว่าง ≥ 10 GB)
   uv run python -c "import torch,platform;print(torch.__version__, platform.release(), torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
   #  ต้องได้ 2.14.0+cu126 ... True NVIDIA GeForce RTX 4060  — ถ้าเป็น +cpu แปลว่า kernel ไม่ใช่ WSL2; ถ้า False แปลว่า driver เก่า
   uv run python -c "import timm; [timm.create_model(m, pretrained=True) for m in ('resnet18','efficientnet_b0')]"   # โหลด pretrained ล่วงหน้า (~70 MB)
   uv run pytest -q tests/test_data.py tests/test_augment.py tests/test_infer.py
   uv run python scripts/run_local_queue.py --list-remaining
   ```
   > เก็บโปรเจกต์ไว้ในดิสก์ของ WSL (`~/work`) ไม่ใช่ `/mnt/c/...` — I/O เร็วกว่าหลายเท่า
4. ติดตั้ง agent: Claude Code (`npm i -g @anthropic-ai/claude-code` หรือตัวติดตั้งของคุณ) และ/หรือ Antigravity CLI (`agy`) แล้ว
   เปิดใน `~/work/Deep_CNN` → agent จะอ่าน `CLAUDE.md`/`AGENTS.md` เอง จากนั้นสั่ง:
   > "อ่าน CLAUDE.md และ tasks/HANDOFF.md แล้วทำต่อตามลำดับใน HANDOFF §3 (เริ่มจาก confound check แล้วค่อย final candidates) ใช้ --set device=cuda --stop-on-error ทุกคิว"
5. ระหว่างทำงาน agent จะ `git commit` + `git push` ผล (`runs/*/metrics.json`, `reports/`, `weights/`) ขึ้น GitHub เอง
   → บนโน้ตบุ๊กแค่ `git pull` ก็เห็นผลและรายงานล่าสุด

## ถ้าใช้ Windows native แทน WSL
ทำได้: ติดตั้ง uv (`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`), แตก tarball ด้วย
`tar -xzf Deep_CNN_handoff_2026-09-19.tar.gz` ใน PowerShell (มี bsdtar ในตัว) ลงที่ `C:\work\Deep_CNN` (อย่าไว้ใน Desktop/Documents ที่ OneDrive sync),
ตั้ง `setx PYTHONUTF8 1` และ `setx HF_HUB_DISABLE_SYMLINKS_WARNING 1`, `git config --global core.autocrlf false`, แล้ว `uv sync`,
`uv run python scripts/run_local_queue.py ... --set device=cuda --stop-on-error` (workers = 0 อัตโนมัติ → ช้ากว่าประมาณการ) — สคริปต์ `.sh`
และ `&&` ใน PowerShell 5.1 ใช้ไม่ได้ ให้พิมพ์ทีละบรรทัด

## เวลาที่คาดบน 4060
resnet18@64 px, 48k ภาพ ≈ 12–15 วิ/epoch → 1 config 20 epochs + TTA ≈ 5–6 นาที → 13 config ที่เหลือ ≈ 1.2 ชม.,
doc-split top-3 + 3 seeds + KD ≈ อีก 1 ชม.

## ถ้าอยากให้ผม (agent บนโน้ตบุ๊ก) ควบคุมเครื่องนั้นจากที่นี่แทน
เปิด OpenSSH server ใน WSL แล้วบอก `user@ip` (LAN/Tailscale) — ผมจะ rsync + รันคิว + ดึงผลกลับเองโดย context ไม่หลุด
