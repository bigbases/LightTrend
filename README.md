# Phase 1 Project

기존 `Project.zip` 구조를 유지하되, **Phase 1 실험(C3 / B4 / B3)** 에 필요한 코드만 바로 쓸 수 있도록 정리한 프로젝트다.

## 포함된 Phase 1 실험
- **C3**: Scale-invariant branch specialization 재평가
- **B4**: DDN vs SAN 잔여 scale heterogeneity 진단
- **B3**: iTransformer 입력 std alignment ablation

## 외부에서 연결해야 하는 대용량 자산
이 프로젝트에는 대용량 파일이 포함되어 있지 않다. 아래 디렉터리는 사용자가 외부 볼륨이나 기존 경로에서 연결해야 한다.

- `dataset/`
- `checkpoints/`
- `station/`
- `station_pre/`

권장 구조는 다음과 같다.

```text
Phase1Project/
  dataset/
    ETT-small/
      ETTh1.csv
      ETTh2.csv
      ETTm1.csv
      ETTm2.csv
    electricity.csv
    exchange_rate.csv
    national_illness.csv
    traffic.csv
    weather.csv
  checkpoints/
  station/
  station_pre/
```

## 핵심 변경점
- 기존 `run.py`에 아래 인자를 추가했다.
  - `--seed`
  - `--result_file`
  - `--analysis_dir`
  - `--station_dir`
  - `--station_pre_dir`
  - `--meanstd_mode`
- 기존 `SAN.py`, `DDN.py`는 건드리지 않고, Phase 1 전용 wrapper로 아래 파일을 추가했다.
  - `normalizers/SAN_phase1.py`
  - `normalizers/DDN_phase1.py`
- Phase 1 실행 스크립트는 `scripts/phase1/` 아래에 모았다.

## 빠른 실행

### 1. C3
```bash
python scripts/phase1/run_c3_specialization_reeval.py   --gpu 0   --checkpoints ./checkpoints   --station_dir ./station   --station_pre_dir ./station_pre
```

출력:
- `logs/phase1/c3_specialization_reeval/c3_specialization_reeval.csv`

### 2. B4
```bash
python scripts/phase1/run_b4_scale_heterogeneity.py
```

출력:
- `logs/phase1/b4_scale_heterogeneity/b4_scale_heterogeneity.csv`
- `logs/phase1/b4_scale_heterogeneity/b4_channel_std_distributions.csv`

### 3. B3
```bash
python scripts/phase1/run_b3_input_std_ablation.py --gpu 0
python scripts/phase1/collect_b3_input_std_ablation.py
```

출력:
- raw: `logs/phase1/b3_input_std_ablation/raw_results.csv`
- organized: `logs/phase1/b3_input_std_ablation/b3_input_std_ablation.csv`

## B3 variant 의미
- `no_norm`
- `mean_only`
- `mean_only_input_std`
- `mean_only_full_std`
- `full`
- `mean_oracle_std`

## 주의
- `mean_only_input_std`는 **입력 정렬용 instance std**를 사용하고, 미래 std prediction은 denorm에 사용하지 않는다.
- `mean_oracle_std`는 평가 시점에 `future_y`로부터 oracle std를 계산한다.
- 기본 설정은 가이드라인의 Phase 1 범위를 따르되, 필요하면 각 스크립트 내부 grid를 좁혀서 먼저 smoke test를 돌리는 것이 안전하다.
