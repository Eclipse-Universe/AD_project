# Q&A Log

프로젝트를 진행하며 나온 "왜?" 질문과 답을 기록한다. `docs/EXPERIMENT_LOG.md`가 "무엇을 시도해서
점수가 어땠는지"를 기록한다면, 이 문서는 "그 과정에서 어떤 개념을 어떻게 이해해 나갔는지"를 기록한다
— 포트폴리오/PPT에서 사고 과정을 보여줄 때 직접 활용할 수 있도록 질문 단위로 정리한다.

---

## 2026-06-30

**Q. src/ 5개 파일을 전부 직접 타이핑해야 하나, 아니면 baseline 코드를 옮겨주면 수정만 하면 되나?**
처음엔 "함수 시그니처+TODO만 스캐폴드, 본문은 직접 작성"을 제안했으나, baseline 로직(데이터 로드,
feature 선택, fit, 라벨 변환)은 이미 셀 단위로 충분히 리뷰를 마친 상태라 다시 타이핑해도 추가
이해가 생기지 않는다는 점에서 "복붙 후 수정" 쪽으로 방향을 바꿨다. 진짜 학습/판단이 필요한 지점은
이식 자체가 아니라 "그 다음에 뭘 바꾸는가"(contamination 등 파라미터)이고, 그건 어차피 직접 한다.
포팅하면서 baseline cell 50-51의 중복 fit 버그는 제거했고, `run_experiment.py` 실행 결과가
`baseline_code/output.csv`와 710,400행 전부 동일함을 직접 diff로 검증했다.

**Q. 계속 실험할 거면 baseline처럼 ipynb로 하는 게 낫지 않나?**
반대 의견을 냈다: `run_experiment.py` 1회 실행이 6.7초로, notebook이 주는 "데이터 재로드 없이
빠르게 반복"이라는 이점이 거의 없다. 반대로 notebook의 단점(cell 상태가 꼬여 의도와 다른 게
실행되는 문제)은 baseline cell 50-51에서 실제로 겪었다. 또 notebook은 git diff가 지저분해 포트폴리오
가독성이 떨어진다. 다만 "결과를 바로 보고 싶다"는 요구는 타당해서, `run_experiment.py` 끝에 label
분포와 predicted positive rate를 바로 출력하는 코드를 추가해 notebook 없이도 즉시 확인 가능하게
했다.

**Q. output 파일이 계속 쌓일 텐데 어떻게 관리해야 하나? (output2, output3...로 명명하려 했음)**
순번 방식(`output2`, `output3`)은 baseline=Exp 0과 번호가 하나씩 밀려 나중에 "output5가 Exp
몇이었지?"를 매번 EXPERIMENT_LOG에서 찾아야 하는 문제가 생긴다고 지적했다. 대신 `outputs/` 디렉토리를
만들고 `EXP_NAME`을 `run_experiment.py` 상수로 둬서 파일명을 `output_exp0.csv`, `output_exp1.csv`
처럼 실험 번호와 그대로 맞췄다. `outputs/`는 `.gitignore` 처리 — 코드+데이터로 결정론적으로 재현
가능하므로, git에 남겨야 할 실제 기록은 raw csv가 아니라 `docs/EXPERIMENT_LOG.md`라고 판단했다.

**Q. 실험마다 py/ipynb 파일을 따로 저장해서 git에 push하는 게 포트폴리오에 좋은 방향일까?**
권하지 않는다고 답했다. 이건 앞서 안 하기로 한 "노트북을 실험마다 v0/v1/v2로 복제"하는 방식과
본질이 같고, 파일이 계속 늘어나며 공통 코드가 매번 중복된다. 대신 `src/`는 항상 최신 실험 1벌만
반영하고, git commit 자체가 그 기록 역할을 하게 했다 — 제출에 쓴 코드만 commit하고, 그 커밋의 짧은
해시를 `docs/EXPERIMENT_LOG.md`에 같이 적어서 `git show <hash>:src/run_experiment.py`로 언제든
복원 가능하게 했다. 포트폴리오 리뷰어에게는 "깔끔한 코드 1벌 + 의미 있는 커밋 히스토리"가 거의 같은
파일 여러 벌보다 더 설득력 있다.

**Q. 현재 Python 3.10.13인데 이걸로 진행해도 괜찮은가?**
직접 확인: 설치된 25개 패키지 버전이 `baseline_code/requirements.txt`와 전부 정확히 일치했고,
이미 이 환경에서 baseline과 동일한 결과를 재현해 실증적으로 검증된 상태였다. 참고로
`requirements.txt`의 `pathlib==1.0.1`은 Python 3.4+ 표준 라이브러리에 이미 포함된 기능의 백포트라
3.10에서는 불필요하지만, 설치돼 있어도 충돌은 없음을 확인했다.

**Q. `run_experiment.py` 실행 결과(label counts, predicted positive rate)는 무슨 의미이고, 다음
실험은 그냥 숫자만 바꿔 실행하면 되나?**
이 실행은 `EXP_NAME`/`CONTAMINATION`을 아직 안 바꿨으므로 새 실험이 아니라 Exp 0 재현이며, 0.1481은
다음 실험과 비교할 "전(before)" 기준점이라고 짚었다. 워크플로우로는: ① 파라미터 수정 → ② 로컬 실행
후 출력 확인 → ③ 합리적이면 그 시점에 git commit(제출할 코드=커밋된 코드를 일치시키기 위함) → ④
서버 제출 → ⑤ 받은 점수를 `docs/EXPERIMENT_LOG.md`에 커밋 해시와 함께 기록 → ⑥ 로그도 commit&push,
순서를 제안했다. 시도할 때마다 commit할 필요는 없고 실제 제출한 버전만 commit하면 된다.

**Q. `contamination`의 정확한 의미와, "0.32로 수렴해야 한다"는 게 정확히 무슨 뜻인가?**
`IsolationForest`의 `contamination`은 `.fit()`한 데이터(train)의 이상치 점수 분포에서 상위 c%를
"정상/이상" 경계로 삼는 threshold 파라미터다. train은 정의상 100% 정상이므로, `contamination=0.32`를
넣는다고 "train의 32%가 실제로 오염됐다"는 뜻이 아니다 — 본래 통계적 의미와 다르게, **threshold를
조절하는 다이얼**로 이 파라미터를 쓰는 것이다. 0.32는 Exp 0의 confusion matrix 역산(F1, accuracy,
예측 positive 개수만으로 TP/FP/FN/TN을 풀어낸 결과)에서 나온 **test 분포의 추정 실제 이상 비율**이며,
"정답"이 아니라 데이터에 근거한 시작점일 뿐이다. 또한 `contamination=0.32`는 train 점수 분포 기준으로
정해지므로, 그 threshold를 test에 적용했을 때 실제 예측 positive 비율이 정확히 32%가 된다는 보장은
없다 — train과 test의 점수 분포가 다를 수 있기 때문이다. 그래서 Exp 1 결과를 받으면 다시 confusion
matrix를 역산해 recall/precision이 어떻게 움직였는지 보고 다음 값을 조정하는 반복 과정으로 본다.

**(버그) `CONTAMINATION = "0.32"`로 두니 `InvalidParameterError`가 났다 — 왜?**
따옴표 때문에 문자열이 됐다. sklearn `IsolationForest`(1.3+)는 `contamination`에 대해
`_validate_params`로 엄격한 타입 검증을 한다 — 허용값은 리터럴 문자열 `'auto'` 또는 `(0.0, 0.5]`
범위의 실제 float뿐이고, 숫자처럼 보이는 문자열을 자동으로 float로 캐스팅해주지 않는다. `EXP_NAME =
"exp0"`일 때 `CONTAMINATION = "auto"`가 통과했던 건 `'auto'`가 정확히 그 특수 문자열 값이었기
때문. 수정: `CONTAMINATION = 0.32` (따옴표 제거).

**Q. Exp 0/1/2로 threshold(contamination)를 다 찍어봤는데, 다음은 무엇이고 왜 그걸 고르나?**
세 점(predicted rate 14.81%/44.36%/36.13%, F1 0.5607/0.5699/0.5812)이 역U자 곡선을 그렸고, 곡선의
정점(Exp 2)에서도 recall 0.6155/precision 0.5506로 둘 다 평범했다 — threshold를 거의 정확히
맞춰도 안 풀리는 한계라는 뜻이라, Tier 1(threshold 튜닝)을 닫고 Tier 2(시간 정보 feature)로
넘어가기로 했다. 첫 feature로 rolling mean/std가 아니라 **diff(직전 시점 대비 변화량)**를 먼저
시도하기로 한 이유: rolling은 window 크기라는 새 하이퍼파라미터가 생겨 "한 번에 하나씩 바꿔
원인을 분리한다"는 지금까지의 실험 원칙에 어긋나고, diff는 하이퍼파라미터 없이 "고장 초반 점진적
드리프트" 가설을 가장 직접적으로 테스트할 수 있다. 설계 시 두 가지를 반드시 지키기로 함: ①
`simulationRun`별로 그룹화한 뒤 diff를 계산해야 run 경계를 넘는 의미 없는 값이 안 생긴다, ② 각
run의 첫 row는 diff가 NaN이 되는데, 그 row를 버리면 제출 파일이 710,400행이 안 돼 채점이 깨지므로
반드시 0으로 채워야 한다(드롭은 선택지가 아님).

---

(다음 질문은 여기에 이어서 추가)
