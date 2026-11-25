# Modeling, in six diagrams

Simplified pictures of the modeling, so the moving pieces are easy to point at. They describe
structure only: no result, figure or performance number appears here (D21 keeps licensed-source
figures out of the repository, and results belong in experiment reports and the registry). The
decisions D1 to D21 are in [the problem statement](problem-statement.md); the claims behind them are
in [the ledger](ledger.md). The system around the models (logger, store, board) is in
[the architecture](architecture.md).

How to read the diagrams:
- Every box names its ticket. A box built on main also names its module, with two short prefixes:
  `japan/` is `src/horsetrack/countries/japan/` and `core/` is `src/horsetrack/core/`. A box whose
  code is not on main says "planned".
- In diagram 2, the tag in each box gives its status: `v0` is the simple first version running in
  shadow now, `built` is another piece on main, `planned` is a ticket not yet landed. Blue, green
  and dashed grey boxes match the tags.
- A solid arrow is a data flow that exists in code on main; a dotted arrow is one a ticket plans.
- Nothing here is adopted. Every model and rule stays in shadow until the owner adopts it through
  diagram 5, and no code places a bet.

Status as of 2026-11-25. When a ticket in a diagram lands, move its box from planned to built and
name its module.

## 1. The decision in one picture

```mermaid
flowchart LR
    HIST["Years of results at their final prices<br/>#48 core/store/build.py"]
    LOGS["Logged odds paths, development days only (D14)<br/>#22 japan/odds_log.py"]
    INFO["What is known at decision time t<br/>as-of views #80 core/store/asof.py<br/>board at t #113 japan/board/snapshot.py"]
    PROB["Outcome half: chance of the outcome given the final price<br/>C2 to C4; v0 is the calibrated market<br/>#114 japan/models/v0/calibration.py"]
    PRICE["Price half: draws of the final pools from the board at t<br/>C1; v0 nowcast<br/>#116 japan/models/v0/nowcast.py"]
    MIX["Average of p · O_final over the draws (D1)<br/>#118 japan/models/v0/shadow_values.py"]
    TEST{"E[p · O_final | info_t] #gt; 1 ?"}
    YES["Worth a bet in principle.<br/>Today: logged in shadow, never staked<br/>#119 japan/shadow/log.py"]
    NO["No value: avoid, the default rung (D19)"]
    HIST --> PROB
    LOGS --> PRICE
    INFO --> PROB
    INFO --> PRICE
    PRICE -->|"each drawn final price"| PROB
    PROB --> MIX
    PRICE --> MIX
    MIX --> TEST
    TEST -->|"yes"| YES
    TEST -->|"no"| NO
```

A bet pays the final odds, which nobody knows until betting closes, so a bet is worth making only
when its chance of winning times the final price is, on average, more than the stake. The chance
comes from models trained on years of results at the price that actually paid; the final price
comes from a forecast, the nowcast, of how the odds will move between now and the close. For each
simulated final price the chance is worked out again and multiplied by that price, and only then
averaged, so a horse whose price tends to shorten when it is going to win is not overvalued
(decision D1).

## 2. The model stack, C0 to C6

```mermaid
flowchart TB
    subgraph L0["C0: what is known at t"]
        ASOF["As-of views and leakage tests<br/>#80 core/store/asof.py<br/>built"]
        BOARD["Board at t and final board<br/>#113 japan/board/snapshot.py<br/>built"]
    end
    subgraph L1["C1: final-price nowcast"]
        NC0["Drift by pool, odds band and horizon<br/>#116 japan/models/v0/nowcast.py<br/>v0"]
        NC1["Structural nowcast for Win and Place #85,<br/>board quality by horizon #86<br/>planned"]
    end
    subgraph L2["C2: per-horse ability and interactions"]
        ABIL["Dynamic rating #91, Plackett-Luce ability #92,<br/>jockey, trainer and track #93,<br/>competition #94, lineage #95<br/>planned"]
    end
    subgraph L3["C3: speed and par"]
        SPEED["Par and expected top-3 time #96,<br/>horse relative speed p_speed #97<br/>planned"]
    end
    subgraph L4["C4: meta-model at the final price, shrunk toward the market (α = 0)"]
        CAL["Calibrated market (D4)<br/>#114 japan/models/v0/calibration.py<br/>v0"]
        RES["Shrunk segment residual<br/>#125 japan/models/v0/residual.py<br/>v0"]
        META["Meta-model #87 and meld of members #98<br/>planned"]
    end
    subgraph L5["C5: Place, Wide and exotics"]
        ORD["Discounted Harville ordering<br/>#115 japan/models/v0/ordering.py<br/>v0"]
        EXO["Exotics against the pool's own prices #99<br/>planned"]
    end
    subgraph L6["C6: decision layer, shadow only"]
        SV["Shadow value over nowcast draws<br/>#118 japan/models/v0/shadow_values.py<br/>v0"]
        SLOG["Shadow log at each grid point<br/>#119 japan/shadow/log.py<br/>#120 japan/shadow/day.py<br/>built"]
        STAKE["Value with our own stake #100,<br/>staking rungs on paper #101<br/>planned"]
    end
    BOARD --> NC0
    NC0 -.->|"replaced by"| NC1
    ASOF -.-> ABIL
    ASOF -.-> SPEED
    CAL --> RES
    CAL --> ORD
    RES -.->|"generalised by"| META
    ABIL -.-> META
    SPEED -.-> META
    META -.-> EXO
    RES --> SV
    ORD --> SV
    NC0 --> SV
    SV --> SLOG
    SV -.-> STAKE
    NC1 -.-> STAKE
    EXO -.-> STAKE

    classDef v0 fill:#dbeafe,stroke:#1d4ed8,color:#0b1f4d
    classDef built fill:#dcfce7,stroke:#15803d,color:#052e16
    classDef planned fill:#f3f4f6,stroke:#6b7280,stroke-dasharray: 5 5,color:#111827
    class NC0,CAL,RES,ORD,SV v0
    class ASOF,BOARD,SLOG built
    class NC1,ABIL,SPEED,META,EXO,STAKE planned
```

The work is split into seven layers, and each is tested both on its own and for what it adds
against the final price. C0 makes sure every model sees only what was known at the decision time;
C1 forecasts the final price; C2 and C3 describe the horses and the race; C4 combines them with the
market, starting from the market alone so that every member has to earn its weight; C5 turns Win
chances into Place, Wide and exotic chances; C6 turns chances and price forecasts into values,
which are logged in shadow and never bet. The v0 boxes are deliberately simple versions that let
the whole chain run end to end on race days while the planned layers are built (epic #73).

## 3. Members feeding the meta-model

```mermaid
flowchart LR
    PP["Past-performance lines as of t<br/>#128 japan/handicap/pp.py"]
    HCP["handicap_score and its factors: speed, pace,<br/>style, class, form, fit, condition, draw<br/>#129 japan/handicap/factors.py"]
    WRK["Workout features, JRDB, private only<br/>#137 japan/handicap/workouts.py"]
    GOING["Going: cushion, moisture, rail setting<br/>#136 japan/jra/going.py<br/>store ingest #90, features #230: planned"]
    BIAS["Within-day track bias<br/>#138, #224 japan/handicap/track_bias.py"]
    RAT["Dynamic rating #91<br/>planned"]
    INT["Interactions: jockey, trainer, track #93,<br/>competition #94, lineage #95<br/>planned"]
    PL["Plackett-Luce ability on the top 3 #92<br/>planned"]
    PAR["Par and expected top-3 time #96<br/>planned"]
    PSPD["Horse relative speed, p_speed #97<br/>planned"]
    MKT["Calibrated-market baseline (D4)<br/>#114 japan/models/v0/calibration.py"]
    META["Meta-model at the final price,<br/>shrunk toward the market (α = 0)<br/>#87, #98: planned"]
    PP --> HCP
    BIAS -.-> HCP
    PAR -.->|"par swap #219"| HCP
    GOING -.->|"rail setting #253"| BIAS
    GOING -.->|"variant prior"| PAR
    PAR -.-> PSPD
    RAT -.-> PL
    INT -.-> PL
    MKT -.->|"offset"| META
    HCP -.-> META
    WRK -.-> META
    RAT -.-> META
    PL -.-> META
    PSPD -.-> META
```

The meta-model is where everything known about the horses meets the market's own view, the final
price. Each box on the left is a member or a set of features, and an arrow means its output is an
input of the box it points to; today only the past-performance lines feeding the handicap factors
run in code, and every other flow is planned. The meta-model starts from the calibrated market and
gives a member weight only when it improves on the final price, so a member that adds nothing ends
up with no weight. Members built on the licensed JRDB data, such as the workout features, run only
in private models on the owner's machine and never publish a figure (D21).

## 4. From Win to Place, Wide and exotics

```mermaid
flowchart TD
    WIN["Win chance p for every runner<br/>C4: #114 v0, #87 planned"]
    ORD["Discounted Harville ordering: second and third<br/>place from p, with fitted discount powers<br/>#115 japan/models/v0/ordering.py"]
    TOP3["Chance of every top-3 finishing order"]
    PLACE["Place: chance of finishing in the paid places,<br/>top 3, or top 2 in small fields"]
    WIDE["Wide: chance that both runners of a pair<br/>finish in the top 3"]
    EXO["Quinella, Exacta, Trio and Trifecta chances<br/>#99 planned"]
    BRK["Place and Wide values at the low and the<br/>high end of the bracket the pool shows<br/>#118 japan/models/v0/shadow_values.py"]
    POOL["Primary baseline: the pool's own final prices (D5)<br/>exotic dividend share #99 planned,<br/>Place and Wide de-vig: owner decision open"]
    DERIV["Secondary baseline: the same ordering fed<br/>the calibrated market alone (D5)"]
    SCORE["Top-3 order log loss and placed-set score<br/>against the pool's baseline (D6)<br/>#69 core/eval/scoring.py"]
    BLEND["Exotic blend: start from the pool's prices,<br/>the model claims only a shrunk residual<br/>#99 planned"]
    FORM["Box, wheel and formation search<br/>#139 japan/betting/formations.py"]
    WIN --> ORD
    ORD --> TOP3
    TOP3 --> PLACE
    TOP3 --> WIDE
    TOP3 -.-> EXO
    PLACE --> BRK
    WIDE --> BRK
    PLACE -.-> SCORE
    WIDE -.-> SCORE
    POOL -.-> SCORE
    DERIV -.-> SCORE
    EXO -.-> BLEND
    POOL -.-> BLEND
    BLEND -.-> FORM
```

Place and Wide pay when a horse finishes near the front, so their chances come from the chance of
every possible top-3 finish, not from the Win chance alone. The ordering model builds those from the
Win chances and adds them up into the chance of each horse placing and of each pair landing in the
top 3. Before the race the pool shows Place and Wide odds as a range, because the payout depends on
which other horses also place, so v0 gives a value at both ends of that range. Every claim in these
pools is judged first against the pool's own final prices and only second against the Win-derived
ordering (D5, D6).

## 5. How a claim earns adoption

```mermaid
flowchart TD
    PRE["Pre-register the family of claims,<br/>frozen by hash before any scoring<br/>#74 core/eval/registry.py"]
    REG["Registry: every run is a record,<br/>failures too, hash-chained<br/>#74 core/eval/registry.py"]
    OOF["Walk-forward out-of-fold fits<br/>on development days (D9, D14)<br/>#70 core/eval/folds.py"]
    BOOT["Log-loss gain over the final price,<br/>meeting bootstrap (D7)<br/>#69 core/eval/scoring.py, core/eval/bootstrap.py"]
    SNOOP["SPA and StepM over the whole family (D17)<br/>#75 core/eval/snooping.py<br/>segment shrinkage #78 core/eval/segment_roi.py"]
    HOLD["Holdout opening: by the owner, once per<br/>family, at its power target (D15)<br/>#71, #126 core/eval/holdout.py"]
    D16["D16 on the holdout: gain interval above 0,<br/>encompassing weight above 0, and the<br/>full-stack gain for a stack member<br/>encompassing test #87 planned"]
    subgraph TRACKS["Betting rules: two tracks (D18)"]
        LA["Loss-avoidance, Family L: loses less than<br/>a stated alternative on the same races,<br/>never claims a profit<br/>#81, #82, #83 planned"]
        PE["Positive-edge, Family P: realised return<br/>above 0 at final dividends on the holdout<br/>#88 planned"]
    end
    OWN["Owner adoption decision,<br/>with an experiment report and the model card"]
    RUNG["Staking rung (D19): avoid, shadow, flat minimum,<br/>flat small, shrunk fractional Kelly<br/>avoid flags #117 japan/board/flags.py<br/>shadow calibration slope #101 planned"]
    PRE --> REG
    REG --> OOF
    OOF --> BOOT
    BOOT --> SNOOP
    SNOOP --> HOLD
    HOLD --> D16
    BOOT -->|"paired, one-sided"| LA
    D16 -->|"plus holdout return"| PE
    D16 --> OWN
    LA --> OWN
    PE --> OWN
    OWN --> RUNG
```

A model or a betting rule is adopted only by the owner, and only after it survives a fixed
sequence of tests written down before the data is looked at. Every attempt is recorded, failures
included, so the number of things tried is always known and the tests can allow for it. Models are
fitted and scored on development days; the locked holdout days are opened by the owner once per
pre-registered family, and a model must beat the final price there with an interval above zero.
Rules that only avoid losses need to lose less than a stated alternative on the same races, rules
that claim a profit must also make money at the final dividends on the holdout, and either kind
climbs the staking rungs one at a time, starting from avoid (D18, D19).

## 6. Timeline of a race day

```mermaid
flowchart TD
    EVE["Before the first race: the card from 00:00 JST,<br/>earlier days' results from the next day on<br/>#80 japan/config/availability.yaml"]
    GO["Going page, used as published by t<br/>#136 japan/jra/going.py"]
    WT["Race-day body weight and same-day going:<br/>labels for now, post − 60 min once<br/>post_utc exists #227 planned"]
    GRID["T-60, T-30, T-16, T-5: grid points,<br/>board at t from the odds log<br/>#24 japan/logger/schedule.py<br/>#113 japan/board/snapshot.py"]
    T2["T-2: the primary decision time (D2),<br/>shadow values logged at every grid point<br/>#120 japan/shadow/day.py"]
    T1["T-1: last grid point, about when<br/>online sales close"]
    POST["Post time: the off, betting closed"]
    FIN["Final odds posted: the price that pays,<br/>a label or conditioner, never an input at t<br/>#113, #234 japan/board/snapshot.py"]
    RES["Result published by JRA, time logged #225"]
    NEXT["Later races that day may use it once published:<br/>within-day bias #138 japan/handicap/track_bias.py"]
    EVE --> GRID
    GO -.->|"if published by t"| GRID
    WT -.->|"not yet"| GRID
    GRID --> T2
    T2 --> T1
    T1 --> POST
    POST --> FIN
    FIN --> RES
    RES --> NEXT
```

Through a race day the odds are captured on a fixed grid of minutes before the off, and at each
grid point the models may use only what had been published by then (the rule for every table is in
[the as-of spec](specs/as-of.md)). T-2 is the main decision time and T-1 is about the last moment a
bet could be placed; the price that actually pays appears only after the off, so before then it can
only be forecast. A race's result can feed later races the same day, for example through the
within-day track bias, but only once JRA has published it.
