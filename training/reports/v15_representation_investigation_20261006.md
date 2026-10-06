# V15 representation investigation — 2026-10-06

## 1. Executive conclusion

Classification **D**. Perform one focused native-maneuver-state observability audit; current evidence does not justify a feature/policy change. No policy or official contract changed.

## 2. Exact 13D contract

Unchanged original columns 0–12: ball_forward, ball_right, ball_up, prediction_forward, prediction_right, prediction_up, ball_distance, car_speed, ball_present, prediction_valid, prediction_horizon, prediction_first_offset, callback_dt.
Car-native forward/right/up geometry; no team inversion. XYZ/distance /6000, speed /2300, masks 0/1, horizon /2 s, first offset /(1/120 s), dt /(1/60 s). Exact selector int((elapsed+2-first_slice_time)*120); no interpolation/clamping. Missing-ball handling unchanged.

## 3. Native field availability audit

See protocol.md Phase 0 table and exact installed SDK/source locations (also embedded in JSON source_audit). V10 records physics and prior submissions; native maneuver flags/packet last_input/boost are absent. No reconstruction is invented.

## 4. 13D ambiguity analysis

| Condition | Query mode accuracy | Pure Jump accuracy | Dodge recall | Boundary accuracy |
|---|---:|---:|---:|---:|
| 13D_H1 | 0.6810551558752997 | 0.20967741935483872 | 0.3380281690140845 | 0.4356929212362911 |
| 13D_A_H1 | 0.6805755395683454 | 0.27419354838709675 | 0.2869718309859155 | 0.4247258225324028 |
| 13D_B_H1 | 0.6896882494004796 | 0.1935483870967742 | 0.3732394366197183 | 0.43469591226321036 |
| 13D_D_H1 | 0.939568345323741 | 0.7741935483870968 | 0.9225352112676056 | 0.8743768693918246 |
| 13D_E_H1 | 0.6498800959232613 | 0.14516129032258066 | 0.2764084507042254 | 0.37886340977068794 |
| 13D_H2 | 0.6729016786570743 | 0.1989247311827957 | 0.31514084507042256 | 0.42273180458624127 |
| 13D_H4 | 0.6748201438848921 | 0.21505376344086022 | 0.3045774647887324 | 0.423728813559322 |
| 13D_H8 | 0.6954436450839329 | 0.3064516129032258 | 0.3397887323943662 | 0.4576271186440678 |
| 13D_H16 | 0.7309352517985611 | 0.26881720430107525 | 0.47007042253521125 | 0.5024925224327019 |
| 13D_A_H16 | 0.708872901678657 | 0.26881720430107525 | 0.3732394366197183 | 0.4715852442671984 |
| 13D_B_H16 | 0.6964028776978417 | 0.23118279569892472 | 0.3714788732394366 | 0.4506480558325025 |
| 13D_D_H16 | 0.9400479616306955 | 0.7741935483870968 | 0.9242957746478874 | 0.8753738783649053 |
| 13D_E_H16 | 0.6426858513189448 | 0.0913978494623656 | 0.301056338028169 | 0.3629112662013958 |

Probe is fixed 32-neighbor voting, not a fitted/deployed policy. Overall representative-grid metrics, distance bins, k=1/8/32/128 label distributions and physical examples are in JSON/diagnostic files.

## 5. Temporal-history analysis

Fixed H=1,2,4,8,16 causal histories, common endpoints; checks: `{"mode_accuracy": false, "pure_jump_accuracy": true, "front_dodge_recall": true, "overall_guard": true}`. Context spans use actual clocks; no future callbacks or match crossing.

## 6. Candidate feature groups

A/B/D tested individually; E evidence gate=True; C unavailable. No combinations or Group F. Candidate checks: `{"A": {"mode_accuracy": false, "pure_jump_accuracy": false, "front_dodge_recall": false, "overall_guard": true, "close_support": true, "close_gain": false, "boundary_support": true, "both_validation_matches": false}, "B": {"mode_accuracy": false, "pure_jump_accuracy": false, "front_dodge_recall": false, "overall_guard": false, "close_support": true, "close_gain": false, "boundary_support": true, "both_validation_matches": false}, "E": {"mode_accuracy": false, "pure_jump_accuracy": false, "front_dodge_recall": false, "overall_guard": false, "close_support": true, "close_gain": false, "boundary_support": true, "both_validation_matches": false}}`.

## 7. Feature sufficiency probe results

Full per-mode/native jump/dodge and boundary/subset metrics are in report.json. Group D gains describe teacher-forced history association only. No validation-driven k/scale/architecture tuning.

## 8. Kickoff-specific findings

| Condition | Validation kickoff rows | Mode accuracy | Pure Jump recall | Dodge recall | V13 kickoff anchor agreements |
|---|---:|---:|---:|---:|---:|
| 13D_H1 | 419 | 0.8949880668257757 | 0.6862745098039216 | 0.8435374149659864 | 3/4 |
| 13D_A_H1 | 419 | 0.9045346062052506 | 0.6862745098039216 | 0.8707482993197279 | 3/4 |
| 13D_B_H1 | 419 | 0.8568019093078759 | 0.6862745098039216 | 0.8163265306122449 | 3/4 |
| 13D_D_H1 | 419 | 0.9260143198090692 | 0.7843137254901961 | 0.9251700680272109 | 0/4 |
| 13D_E_H1 | 419 | 0.837708830548926 | 0.5294117647058824 | 0.7619047619047619 | 2/4 |
| 13D_H2 | 419 | 0.883054892601432 | 0.6862745098039216 | 0.8095238095238095 | 3/4 |
| 13D_H4 | 419 | 0.9021479713603818 | 0.7843137254901961 | 0.8367346938775511 | 3/4 |
| 13D_H8 | 419 | 0.9212410501193318 | 0.8431372549019608 | 0.8707482993197279 | 3/4 |
| 13D_H16 | 419 | 0.9355608591885441 | 0.8431372549019608 | 0.9047619047619048 | 3/4 |
| 13D_A_H16 | 419 | 0.9379474940334129 | 0.8431372549019608 | 0.8979591836734694 | 3/4 |
| 13D_B_H16 | 419 | 0.8878281622911695 | 0.8431372549019608 | 0.8503401360544217 | 3/4 |
| 13D_D_H16 | 419 | 0.9260143198090692 | 0.7843137254901961 | 0.9251700680272109 | 0/4 |
| 13D_E_H16 | 419 | 0.837708830548926 | 0.3333333333333333 | 0.8707482993197279 | 0/4 |

Shadow anchor labels are diagnostic on learner-induced physics, not approved training corrections. Passing this probe is not closed-loop maneuver success.

## 9. Recovery/inversion-specific findings

| Condition | Inverted rows / accuracy | Tilted rows / accuracy | Region rows / accuracy | Recovery anchor agreements |
|---|---:|---:|---:|---:|
| 13D_H1 | 324 / 0.7191358024691358 | 828 / 0.6594202898550725 | 869 / 0.5822784810126582 | 4/7 |
| 13D_A_H1 | 324 / 0.7716049382716049 | 828 / 0.6268115942028986 | 869 / 0.5592635212888377 | 4/7 |
| 13D_B_H1 | 324 / 0.7160493827160493 | 828 / 0.6835748792270532 | 869 / 0.6052934407364787 | 1/7 |
| 13D_D_H1 | 324 / 0.9444444444444444 | 828 / 0.961352657004831 | 869 / 0.9321058688147296 | 1/7 |
| 13D_E_H1 | 324 / 0.7098765432098766 | 828 / 0.6171497584541062 | 869 / 0.5845799769850403 | 5/7 |
| 13D_H2 | 324 / 0.7037037037037037 | 828 / 0.6557971014492754 | 869 / 0.5753739930955121 | 4/7 |
| 13D_H4 | 324 / 0.7006172839506173 | 828 / 0.6533816425120773 | 869 / 0.570771001150748 | 5/7 |
| 13D_H8 | 324 / 0.7623456790123457 | 828 / 0.6618357487922706 | 869 / 0.5937859608745685 | 5/7 |
| 13D_H16 | 324 / 0.7901234567901234 | 828 / 0.716183574879227 | 869 / 0.6421173762945915 | 6/7 |
| 13D_A_H16 | 324 / 0.7654320987654321 | 828 / 0.6521739130434783 | 869 / 0.6018411967779056 | 4/7 |
| 13D_B_H16 | 324 / 0.7191358024691358 | 828 / 0.6751207729468599 | 869 / 0.6248561565017261 | 1/7 |
| 13D_D_H16 | 324 / 0.9444444444444444 | 828 / 0.9625603864734299 | 869 / 0.9332566168009206 | 1/7 |
| 13D_E_H16 | 324 / 0.6790123456790124 | 828 / 0.6147342995169082 | 869 / 0.571921749136939 | 6/7 |

Native world-up alignment, inverted/tilted subsets and query-anchor physical metadata distinguish geometry from contact/maneuver prerequisites. Long-Neutral subset in teacher demonstrations is Neutral-label context, not evidence that this teacher was stuck. Known V13 long-pause windows are shadow-label queries only. Field proximity is not wall collision.

## 10. Representation versus temporal classification

D; finite observational criteria, not proof. Native C data unavailability limits this classification.

## 11. One recommended next experiment

Perform one focused native-maneuver-state observability audit; current evidence does not justify a feature/policy change

## 12. Experiments NOT authorized

No full policy training, test evaluation, live game, DAgger, deployment, official observation/architecture/teacher change.

## 13. Integrity/hash results

All authorized inputs hash-verified before/after; 43 protected hashes unchanged. Frozen V10 and V13/V14 inputs remain unchanged.

## 14. Exact artifacts read

report.json artifacts_read/source_seal lists paths and SHA256s. Native source-only audit preceded all sample analysis.

## 15. Limitations

- Finite histories do not exhaust GRU memory or prove observability/insufficiency
- H16 is shorter than the complete original maneuver; representation-versus-horizon classification is exploratory
- Original teacher has no explicit recovery policy; predicting its labels is not optimal-recovery evidence
- Neighbor scale and correlated callbacks affect purity; no significance or causal claim
- Diagnostic query subset, not full validation population; match/opponent overlap remains
- Group D is prior submitted teacher controls, not packet last_input or student-feedback robustness
- Native maneuver flags/boost not recorded in V10; Group C cannot be compared
- Shadow query labels on learner physics have unqualified maneuver prerequisites
- Region proximity does not prove wall contact; no synchronized video
- Double-precision norm/dot search has rounding; returned distances recomputed directly

STOP for review.
