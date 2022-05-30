from bidi.algorithm import get_display
import numpy as np
import cv2
import torch
import os
import sys
from PIL import Image
from logging import getLogger
import yaml

if sys.version_info[0] == 2:
    from io import open
    from six.moves.urllib.request import urlretrieve
    from pathlib2 import Path
else:
    from urllib.request import urlretrieve
    from pathlib import Path

# print(os.environ)
# # environ({'CONDA_SHLVL': '2', 'LS_COLORS': 'rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.Z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.zst=01;31:*.tzst=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.wim=01;31:*.swm=01;31:*.dwm=01;31:*.esd=01;31:*.jpg=01;35:*.jpeg=01;35:*.mjpg=01;35:*.mjpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.m4a=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.oga=00;36:*.opus=00;36:*.spx=00;36:*.xspf=00;36:', 'LD_LIBRARY_PATH': ':/usr/local/cuda-11.1/lib64', 'CONDA_EXE': '/home/user/anaconda3/bin/conda', 'LC_MEASUREMENT': 'zh_CN.UTF-8', 'SSH_CONNECTION': '115.156.248.247 50537 192.168.1.198 22', 'LESSCLOSE': '/usr/bin/lesspipe %s %s', 'LC_PAPER': 'zh_CN.UTF-8', 'LC_MONETARY': 'zh_CN.UTF-8', 'LANG': 'en_US.UTF-8', 'COLORTERM': 'truecolor', 'CONDA_PREFIX': '/home/user/anaconda3/envs/dr_ocr', 'VSCODE_GIT_ASKPASS_EXTRA_ARGS': '', '_CE_M': '', 'LC_NAME': 'zh_CN.UTF-8', 'XDG_SESSION_ID': '427', 'USER': 'user', 'CONDA_PREFIX_1': '/home/user/anaconda3', 'PWD': '/home/user/xiongdengrui/ocr', 'HOME': '/home/user', 'CONDA_PYTHON_EXE': '/home/user/anaconda3/bin/python', 'BROWSER': '/home/user/.vscode-server/bin/da15b6fd3ef856477bf6f4fb29ba1b7af717770d/bin/helpers/browser.sh', 'VSCODE_GIT_ASKPASS_NODE': '/home/user/.vscode-server/bin/da15b6fd3ef856477bf6f4fb29ba1b7af717770d/node', 'TERM_PROGRAM': 'vscode', 'SSH_CLIENT': '115.156.248.247 50537 22', 'TERM_PROGRAM_VERSION': '1.67.1', 'XDG_DATA_DIRS': '/usr/local/share:/usr/share:/var/lib/snapd/desktop', '_CE_CONDA': '', 'VSCODE_IPC_HOOK_CLI': '/run/user/1000/vscode-ipc-7594c603-b830-4186-b13d-1bd1516f86c0.sock', 'LC_ADDRESS': 'zh_CN.UTF-8', 'LC_NUMERIC': 'zh_CN.UTF-8', 'CONDA_PROMPT_MODIFIER': '(dr_ocr) ', 'MAIL': '/var/mail/user', 'VSCODE_GIT_ASKPASS_MAIN': '/home/user/.vscode-server/bin/da15b6fd3ef856477bf6f4fb29ba1b7af717770d/extensions/git/dist/askpass-main.js', 'TERM': 'xterm-256color', 'SHELL': '/bin/bash', 'SHLVL': '4', 'VSCODE_GIT_IPC_HANDLE': '/run/user/1000/vscode-git-5e06f80b3c.sock', 'LC_TELEPHONE': 'zh_CN.UTF-8', 'LOGNAME': 'user', 'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/1000/bus', 'GIT_ASKPASS': '/home/user/.vscode-server/bin/da15b6fd3ef856477bf6f4fb29ba1b7af717770d/extensions/git/dist/askpass.sh', 'XDG_RUNTIME_DIR': '/run/user/1000', 'PATH': '/home/user/.vscode-server/bin/da15b6fd3ef856477bf6f4fb29ba1b7af717770d/bin/remote-cli:/home/user/anaconda3/envs/dr_ocr/bin:/home/user/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda-11.1/bin', 'LC_IDENTIFICATION': 'zh_CN.UTF-8', 'CONDA_DEFAULT_ENV': 'dr_ocr', 'LESSOPEN': '| /usr/bin/lesspipe %s', 'LC_TIME': 'zh_CN.UTF-8', '_': '/home/user/anaconda3/envs/dr_ocr/bin/python'})

# MODULE_PATH = os.environ.get("EASYOCR_MODULE_PATH") or \
#               os.environ.get("MODULE_PATH") or \
#               os.path.expanduser("~/.EasyOCR/")
# print(MODULE_PATH)
# # /home/user/.EasyOCR/

# printed_list = ['standard'] + [model for model in {'1': [1], '2': [2]}] + [model for model in {'3': [3], '4': [4]}]
# print(printed_list)

# BASE_PATH = os.path.dirname(__file__)
# # print(BASE_PATH)

# dict_list = {}
# # BASE_PATH = /home/user/xiongdengrui/ocr/EasyOCR
# for lang in ['ch_sim','en']:
#     dict_list[lang] = os.path.join(BASE_PATH, 'dict', lang + ".txt")
    
# print(dict_list)

# 0 ([[37, 19], [513, 19], [513, 61], [37, 61]], '疫情防控不放松,这六要点要牢记', 0.6488854047606668)
# 1 ([[41, 59], [203, 59], [203, 101], [41, 101]], '中国这十年', 0.6731725025639657)
# 2 ([[241, 59], [651, 59], [651, 101], [241, 101]], '中国经济实力迈上一个大台阶', 0.5842685247424994)
# 3 ([[39, 99], [623, 99], [623, 141], [39, 141]], '[奋斗者正青春]以奋斗之姿  冲顶世界之巅', 0.9230386639973365)
# 4 ([[49, 138], [638, 138], [638, 182], [49, 182]], '4美国挑唆俄乌冲突意味着越来越大的危险', 0.3044573827346284)
# 5 ([[41, 181], [671, 181], [671, 223], [41, 223]], '专题 |强国复兴有我  奋进新征程  建功新时代', 0.6305919214147327)
# 6 ([[39, 221], [247, 221], [247, 263], [39, 263]], '国务院办公厅:', 0.7369343042116138)
# 7 ([[257, 221], [677, 221], [677, 263], [257, 263]], '2023年起不再发放就业报到证', 0.9659873509026033)
# 8 ([[102, 456], [734, 456], [734, 584], [102, 584]], 'Baiew百度', 0.5008282063903912)
# 9 ([[58, 676], [106, 676], [106, 700], [58, 700]], '止海:', 0.5255587740854731)
# 10 ([[414, 676], [712, 676], [712, 704], [414, 704]], '要努力在5月中旬实现社会面清零', 0.8682800390790403)
# 11 ([[54, 758], [320, 758], [320, 788], [54, 788]], '31省区市新增本土 "253+1726', 0.9267344285760961)
# 12 ([[411, 756], [632, 756], [632, 786], [411, 786]], '新增1例死亡病例在上海', 0.7866887480310617)
# 13 ([[58, 840], [354, 840], [354, 868], [58, 868]], '上海昨日新增本土确诊病例194例_', 0.4186134069651198)
# 14 ([[414, 840], [642, 840], [642, 866], [414, 866]], '本土无症状感染者148~  例', 0.43381789228530393)
# 15 ([[53, 920], [146, 920], [146, 950], [53, 950]], '天津通报:', 0.9392877224448832)
# 16 ([[414, 922], [634, 922], [634, 952], [414, 952]], '发现2名核酸阳性感染者', 0.9672670026863118)
# 17 ([[772, 922], [994, 922], [994, 952], [772, 952]], '全市将开展全员核酸检测', 0.9182460703169596)
# 18 ([[54, 1006], [126, 1006], [126, 1032], [54, 1032]], '外交部:', 0.9932913184165955)
# 19 ([[414, 1006], [622, 1006], [622, 1032], [414, 1032]], '美方执意邀请台湾方面,', 0.4295193864203894)
# 20 ([[772, 1006], [1012, 1006], [1012, 1034], [772, 1034]], '中方无法出席全球抗疫峰会', 0.9552888072096967)
# 21 ([[54, 1088], [264, 1088], [264, 1116], [54, 1116]], '茅台董事长高卫东落马,', 0.8170332465015812)
# 22 ([[414, 1087], [674, 1087], [674, 1115], [414, 1115]], '曾违规披露信息被证监会点名', 0.6379641842837281)
# 23 ([[39, 1180], [639, 1180], [639, 1223], [39, 1223]], '中方是否有从朝鲜撤侨的计划?  外交部回应', 0.9309378628501764)
# 24 ([[37, 1262], [669, 1262], [669, 1305], [37, 1305]], '埃尔多安:  土耳其不支持瑞典和芬兰加入北约', 0.8470623167041773)
# 25 ([[37, 1343], [607, 1343], [607, 1387], [37, 1387]], '韩国总统尹锡悦:  愿向朝鲜提供医疗物资', 0.5055362064107709)
# 26 ([[38, 1422], [840, 1422], [840, 1470], [38, 1470]], '印度新德里一建筑突发火灾已致数十人死亡  莫迪表示哀悼', 0.8790266863211823)
# 27 ([[39, 1505], [871, 1505], [871, 1549], [39, 1549]], '以色列警察与被枪杀女记者送葬队伍发生冲突  棺材差点落地', 0.7947388460471694)
# 28 ([[37, 1587], [731, 1587], [731, 1629], [37, 1629]], '妆容精致气色佳! 英国女王公开露面皇家温莎马展', 0.607247825055156)

