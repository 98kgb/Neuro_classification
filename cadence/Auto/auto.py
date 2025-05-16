#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code optimizes circuit parameters for EEG classification
and extracts simulation results using Spectre and Ocean.

@author: gibaek
"""
import os, sys
import subprocess

# 경로 설정
path_c = os.path.dirname(os.path.abspath(__file__))
path_p = os.path.dirname(path_c)
path_g = os.path.dirname(path_p)
root = os.path.dirname(os.path.dirname(path_g))
sys.path.append(path_c)

# netlist에 파라미터 삽입
def update_netlist(params: dict, template_path="template.scs", output_path="run.scs"):
    with open(template_path, "r") as f:
        text = f.read()
    for key, value in params.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    with open(output_path, "w") as f:
        f.write(text)

# netlist에 save 명령 추가
def insert_save_statement(scs_path, save_signals, output_path=None):
    with open(scs_path, 'r') as f:
        lines = f.readlines()

    save_line = "save " + " ".join(save_signals) + "\n"

    insert_idx = -1
    for idx, line in enumerate(lines):
        if 'saveOptions' in line:
            insert_idx = idx + 1
            break

    if insert_idx != -1:
        lines.insert(insert_idx, save_line)
    else:
        print("❗ Warning: 'saveOptions' not found. Appending save line at end.")
        lines.append("\n" + save_line)

    out_path = output_path if output_path else scs_path
    with open(out_path, 'w') as f:
        f.writelines(lines)

    print(f"✅ save 문 추가 완료: {save_line.strip()} → {out_path}")

    # 디버깅용: 저장 내용 다시 출력
    print("=== Netlist 일부 미리보기 ===")
    for line in lines[-10:]:
        print(line.strip())


# spectre 시뮬레이션 실행
def run_spectre(netlist_path="run.scs", result_dir=f"{path_c}/results/C0_3p/psf", log_path="log.txt"):
    os.makedirs(result_dir, exist_ok=True)  # 결과 폴더 없으면 생성
    result = subprocess.run(["spectre", "-raw", result_dir, netlist_path],
                            capture_output=True, text=True)

    with open(f"{path_c}/spectre_stdout.txt", "w") as f:
        f.write(result.stdout)
    with open(f"{path_c}/spectre_stderr.txt", "w") as f:
        f.write(result.stderr)

    print(f"✅ Spectre 실행 완료. 결과 저장 위치: {result_dir}")
    if result.returncode != 0:
        print("❌ 에러 발생! stderr 출력:")
        print(result.stderr)


# ocean 스크립트 생성
def generate_ocean_script(signal="Vout_T8-P8", result_dir="psf_output", output_csv="Vout_T8-P8.csv", script_path="extract.ocn"):
    script = f"""
                openResults(\"{result_dir}\")
                selectResult('tran)
                
                printf(\"=== Available signals ===\\n\")
                foreach(sig outputs()
                  printf(\"%s\\n\" sig)
                )
                
                let((vout timeVals valVals fp)
                  vout = v(\"{signal}\")
                  if(sprintf(nil \"%s\" type(vout)) == \"waveform\" then
                    timeVals = xval(vout)
                    valVals = yval(vout)
                
                    fp = outfile(\"{output_csv}\" \"w\")
                    for(i 0 (length(timeVals) - 1)
                      fprintf(fp \"%g,%g\\n\" (nth(i timeVals)) (nth(i valVals)))
                    )
                    close(fp)
                  )
                )
                """
                
    with open(script_path, "w") as f:
        f.write(script)
    print(f"✅ Ocean 스크립트 생성 완료: {script_path}")

# ocean 실행
def run_ocean(script_path="extract.ocn"):
    result = subprocess.run(["ocean", "-nograph", "-restore", script_path],
                            capture_output=True, text=True)
    print("=== OCEAN STDOUT ===")
    print(result.stdout)
    print("=== OCEAN STDERR ===")
    print(result.stderr)

    if result.returncode != 0:
        print("❌ OCEAN 실행 오류 발생!")

#%% 메인 실행 파트
lib_name = 'GB'
cell_name = 'AH_circuit_multi_ch'

in_path = f'{path_c}/base.scs'
out_path = f'{path_c}/input.scs'

# 파라미터와 저장 신호 설정
params = {'C0': '3p'}
save_signals = ["Vout_T8-P8", "Vmem_T8-P8"]

update_netlist(params=params, template_path=in_path, output_path=out_path)
insert_save_statement(scs_path=out_path, save_signals=save_signals)
run_spectre(netlist_path=out_path)
#%%
generate_ocean_script(
    signal="Vout_T8-P8",
    result_dir=f"{path_c}/results/C0_3p/psf",
    output_csv="{path_c}/C0_3p.csv",
    script_path = f"{path_c}/extract.ocn"
)
# generate_ocean_script(signal="Vout_T8-P8", result_dir="psf_output", output_csv="Vout_T8-P8.csv")
#%%
run_ocean()
