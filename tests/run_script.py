import argparse
import itertools
import subprocess

# 가능한 인수 값들 정의
lr_values = [1e-5, 1e-6]
update_mode_values = [0, 1, 2, 3, 4, 5]
text_emb_values = [0, 1]
usage_classifier_values = [0, 1]

# 가능한 조합 생성
combinations = list(itertools.product(lr_values, update_mode_values, text_emb_values, usage_classifier_values))

# 조합별로 스크립트 실행
for combo in combinations:
    lr, update_mode, text_emb, usage_classifier = combo

    # 명령어 생성
    cmd = f"python clip_train.py -lr {lr} -update_mode {update_mode} -text_emb {text_emb} -usage_clssifier {usage_classifier}"

    # 스크립트 실행
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # 스크립트 실행 결과 출력
    print(f"Command: {cmd}")
    print(f"Exit Code: {result.returncode}")
    print(f"Output: {result.stdout}")
    print(f"Error: {result.stderr}")