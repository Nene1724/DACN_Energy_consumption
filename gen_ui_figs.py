"""
Generate UI-accurate paper figures:
  Fig 4  → Paper/figures/17_web_deploy_monitoring.png
  Fig 15 → Paper/figures/fig16_medical_monitoring.png
"""
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

OUT = "Paper/figures"
os.makedirs(OUT, exist_ok=True)

# ── colour palette (from project CSS) ──────────────────────────────
BG       = '#f5f7fb'
PANEL    = '#ffffff'
ACCENT   = '#3b82f6'
SUCCESS  = '#16a34a'
DANGER   = '#dc2626'
WARNING  = '#f59e0b'
TEXT     = '#0b1b32'
MUTED    = '#64748b'
BORDER   = '#e2e8f0'
CYAN     = '#22d3ee'
DARK     = '#020617'

DPI = 180

# ── helpers ────────────────────────────────────────────────────────
def card(ax, x, y, w, h, fc=PANEL, ec=BORDER, r=0.015, shadow=True, zorder=2):
    if shadow:
        ax.add_patch(FancyBboxPatch((x+.003, y-.003), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            linewidth=0, facecolor='#00000015',
            transform=ax.transAxes, zorder=zorder-1))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=.8, edgecolor=ec, facecolor=fc,
        transform=ax.transAxes, zorder=zorder))

def txt(ax, x, y, s, fs=9, color=TEXT, ha='left', va='center',
        fw='normal', zo=10, **kw):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
            fontweight=fw, transform=ax.transAxes, zorder=zo, **kw)

def pill(ax, x, y, w, h, label, fc, tc, fs=6.5, zo=6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=.004",
        linewidth=0, facecolor=fc,
        transform=ax.transAxes, zorder=zo))
    txt(ax, x+w/2, y+h/2, label, fs=fs, color=tc, ha='center', fw='bold', zo=zo+1)

def btn(ax, x, y, w, h, label, fc, tc, ec=None, fs=8, zo=5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=.007",
        linewidth=.8 if ec else 0,
        edgecolor=ec or fc, facecolor=fc,
        transform=ax.transAxes, zorder=zo))
    txt(ax, x+w/2, y+h/2, label, fs=fs, color=tc, ha='center', fw='bold', zo=zo+1)


# ══════════════════════════════════════════════════════════════════
# FIG 4 — Main deployment dashboard  (12 × 7 in)
# ══════════════════════════════════════════════════════════════════
W4, H4 = 12, 7
fig, ax = plt.subplots(figsize=(W4, H4))
ax.set(xlim=(0,1), ylim=(0,1)); ax.axis('off')
ax.set_facecolor(BG)
ax.add_patch(Rectangle((0,0),1,1,facecolor=BG,zorder=0))

# ── title strip ───────────────────────────────────────────────────
ax.plot(.046, .958, 'o', color=CYAN, ms=10, zorder=10, markeredgecolor='none')
txt(ax, .065, .958, 'IoT ML Energy Manager', fs=18, fw='bold', color=TEXT)
txt(ax, .5, .930, 'Energy-Aware OTA Deployment  ·  Runtime Telemetry  ·  Adaptive Management',
    fs=8.5, color=MUTED, ha='center')

# ── nav tabs (5) ──────────────────────────────────────────────────
tabs = [('System',True),('Medical',False),('Deployment',False),
        ('Analytics',False),('Monitoring',False)]
tx = .055
for name, active in tabs:
    ax.add_patch(FancyBboxPatch((tx,.895),.115,.030,
        boxstyle="round,pad=0,rounding_size=.010",
        linewidth=1.8 if active else .7,
        edgecolor=ACCENT if active else BORDER,
        facecolor=PANEL if active else 'none',
        transform=ax.transAxes, zorder=3))
    txt(ax, tx+.0575, .910, name, fs=8.5,
        color=ACCENT if active else MUTED,
        fw='bold' if active else 'normal', ha='center')
    tx += .135

# ── LEFT PANEL  (x: .020–.390, y: .095–.875) ─────────────────────
PL, PR, PB, PT = .020, .390, .095, .875
card(ax, PL, PB, PR-PL, PT-PB)

txt(ax, .038, .848, 'Quick Deploy', fs=12, fw='bold')
txt(ax, .038, .828, 'Select model & device · predict energy · deploy via OTA',
    fs=7.5, color=MUTED)

# form fields — label_y positions; field box: [label_y-.040, label_y-.012]
fields = [
    ('Model',               'MobileNetV3-Small-0.75.tflite',        .790),
    ('Device IP',           'Jetson Nano  ·  192.168.1.105 (Online)',.725),
    ('Energy Budget (mWh)', '15.0',                                   .660),
    ('Canary Duration (s)', '5.0',                                    .597),
]
for label, val, fy in fields:
    txt(ax, .036, fy, label, fs=7.5, color=TEXT, fw='600')
    card(ax, .028, fy-.040, .354, .028, fc='#fdfdff', ec=BORDER, shadow=False, zorder=4)
    txt(ax, .040, fy-.025, val, fs=8, color=TEXT, zo=5)

# buttons  (y: .528–.560, below last field bottom .597-.040=.557 → .560 touches boundary)
# push buttons to y=.520
btn(ax, .028, .520, .168, .032, 'Predict Energy', ACCENT,  'white', fs=8.5)
btn(ax, .210, .520, .168, .032, 'Deploy OTA',     PANEL, TEXT, ec=BORDER, fs=8.5)

# energy prediction result box  (y: .340–.508)
card(ax, .028, .340, .354, .168, fc='#eef6ff', ec='#bfdbfe', shadow=False, zorder=4)
txt(ax, .036, .495, 'Energy Prediction Result', fs=8.5, color=ACCENT, fw='bold', zo=5)
rows = [
    ('Predicted Energy',    '2.91 mWh',            SUCCESS),
    ('Conf. Interval',      '±0.38 mWh',           TEXT),
    ('Recommendation',      'Excellent (in budget)', SUCCESS),
    ('Model R²  |  Pearson r', '0.955  |  0.983',  ACCENT),
]
for i,(lbl,val,col) in enumerate(rows):
    ry = .470 - i*.038
    txt(ax, .037, ry, lbl, fs=7.5, color=MUTED, fw='600', zo=5)
    txt(ax, .260, ry, val, fs=8,   color=col, fw='bold' if col!=TEXT else 'normal', zo=5)

# recent activity  (y: .118–.330)
txt(ax, .036, .322, 'Recent Activity', fs=8.5, fw='bold')
logs = [
    (SUCCESS, 'MobileNetV3-S-0.75 → Jetson Nano (2.91 mWh)'),
    (ACCENT,  'Budget check passed  [2.91 < 15.0 mWh]'),
    (SUCCESS, 'LCNet-050 → RPi5  (1.23 mWh)'),
    (DANGER,  'ResNet50 rollback  — latency exceeded'),
]
for i,(col,msg) in enumerate(logs):
    ry = .295 - i*.040
    ax.plot(.038, ry, 'o', color=col, ms=5, transform=ax.transAxes, zorder=5)
    txt(ax, .052, ry, msg, fs=7.5, color=MUTED)

# ── RIGHT PANEL  (x: .410–.985, y: .095–.875) ────────────────────
RL, RR, RB, RT = .410, .985, .095, .875
card(ax, RL, RB, RR-RL, RT-RB)

txt(ax, .428, .848, 'Model Library', fs=12, fw='bold')
txt(ax, .428, .828, '360 benchmarked models  ·  Jetson Nano  ·  sorted by energy',
    fs=7.5, color=MUTED)

# energy summary cards  (y: .793–.820)
ecards = [('Min','2.39 mWh',SUCCESS),('Median','98.6 mWh',ACCENT),
          ('Mean','324 mWh',WARNING),('Max','11,736 mWh',DANGER)]
for i,(lbl,val,col) in enumerate(ecards):
    ex = .418 + i*.143
    card(ax, ex, .793, .135, .028, fc='#fafafa', ec=BORDER, shadow=False, zorder=4)
    txt(ax, ex+.0675, .818, lbl, fs=6.5, color=MUTED, ha='center', zo=5)
    txt(ax, ex+.0675, .803, val, fs=7.5, color=col,  ha='center', fw='bold', zo=5)

# model list  (7 items; my starts at .750, top=.750+.046=.796 ≤ .793? No → .748)
# my+.046 ≤ .793  →  my ≤ .747  → use .743
# step = (.743-.095-.046)/6 = .084   → (.743-.095-6*.086-.046=.097>0 ✓)
models = [
    ('LCNet-050',          '2.39 mWh','31.0 ms','0.923',SUCCESS,'Recommended'),
    ('MobileNetV3-S-0.75', '2.91 mWh','35.1 ms','0.712',SUCCESS,'Recommended'),
    ('MobileNetV3-S-1.0',  '3.14 mWh','38.4 ms','0.731',SUCCESS,''),
    ('TF-MobileNetV3-Min', '3.90 mWh','24.0 ms','0.635',SUCCESS,''),
    ('MobileViT-XXS',      '7.83 mWh','85.2 ms','0.782',WARNING,''),
    ('EfficientNet-B0',    '12.4 mWh','142 ms', '0.772',WARNING,''),
    ('ResNet18',           '29.4 mWh','347 ms', '0.697',DANGER, ''),
]
my = .740
STEP = .088
for name,energy,lat,acc,col,bdg in models:
    is_ok = col==SUCCESS
    card(ax, .415, my-.012, .562, .052,
         fc='#f0f9ff' if is_ok else PANEL,
         ec='#93c5fd' if is_ok else BORDER, shadow=False, zorder=4)
    txt(ax, .425, my+.026, name,   fs=9,   color=TEXT, fw='bold', zo=5)
    txt(ax, .425, my+.006, f'Energy: {energy}   Lat: {lat}   Top-1: {acc}',
        fs=7, color=MUTED, zo=5)
    if bdg:
        pill(ax, .640, my+.016, .090, .018, bdg, '#dcfce7', SUCCESS, fs=5.8)
    # energy bar
    bx, bw = .752, .092
    ax.add_patch(FancyBboxPatch((bx, my+.010), bw, .009,
        boxstyle="round,pad=0,rounding_size=.002",
        linewidth=0, facecolor='#e2e8f0', transform=ax.transAxes, zorder=5))
    frac = min(float(energy.split()[0])/30,1.0)
    ax.add_patch(FancyBboxPatch((bx, my+.010), bw*frac, .009,
        boxstyle="round,pad=0,rounding_size=.002",
        linewidth=0, facecolor=col, alpha=.85, transform=ax.transAxes, zorder=6))
    btn(ax, .860, my+.004, .093, .026, 'Deploy', PANEL, ACCENT, ec=BORDER, fs=7.5)
    my -= STEP

fig.savefig(f'{OUT}/17_web_deploy_monitoring.png',
            dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.close()
print("OK: Fig 4 saved")


# ══════════════════════════════════════════════════════════════════
# FIG 15 — Medical monitoring  (12 × 7 in)
# ══════════════════════════════════════════════════════════════════
WM, HM = 12, 7
fig, ax = plt.subplots(figsize=(WM, HM))
ax.set(xlim=(0,1), ylim=(0,1)); ax.axis('off')
ax.set_facecolor(BG)
ax.add_patch(Rectangle((0,0),1,1,facecolor=BG,zorder=0))

# ── HEADER (y: .910–.990) ─────────────────────────────────────────
card(ax, .010, .910, .980, .080)
txt(ax, .025, .960, 'Medical Safety Watch', fs=14, fw='bold')
txt(ax, .025, .936, 'Clinical device:  Jetson Nano — 192.168.1.105  |  Last sync: 14:32:07',
    fs=8, color=MUTED)
btn(ax, .740, .922, .240, .034, 'Clinical Watch: Active', ACCENT, 'white', fs=8.5)

# ── STATUS BANNER (y: .848–.904) ──────────────────────────────────
card(ax, .010, .848, .980, .056, fc='#f0fdf4', ec='#86efac', shadow=False, zorder=3)
ax.plot(.030, .876, 's', color=SUCCESS, ms=10, transform=ax.transAxes, zorder=5)
# title on left
txt(ax, .050, .882, 'System Ready — Patient Safety Monitoring',
    fs=10.5, fw='bold', color=SUCCESS, zo=5)
txt(ax, .050, .860, 'Clinical Watch active. Patient posture: Normal. No fall events detected.',
    fs=7.5, color='#166534', zo=5)
# status chips on right (4 chips, right-aligned)
schips = [('State','Monitoring'),('Severity','Normal'),('Watch','Active'),('Score','0.21')]
for i,(k,v) in enumerate(schips):
    txt(ax, .527+i*.122, .876, f'{k}: {v}', fs=7.5, color='#166534', fw='bold', zo=5)

# ── CAMERA PANEL  (x:.010–.570, y:.410–.840) ─────────────────────
CL,CR,CB,CT = .010,.570,.410,.840
card(ax, CL, CB, CR-CL, CT-CB)
txt(ax, .026, .820, 'Live Patient Camera', fs=10, fw='bold')
for chip,cx in [('USB Camera',.225),('MoveNet',  .355),('Jetson Edge',.456)]:
    pill(ax, cx-.010, .811, .096, .016, chip, '#dbeafe', ACCENT, fs=6)

# camera dark viewport  (y:.420–.797)
VL,VR,VB,VT = .018,.560,.420,.800
ax.add_patch(FancyBboxPatch((VL,VB),VR-VL,VT-VB,
    boxstyle="round,pad=0,rounding_size=.006",
    linewidth=0, facecolor=DARK, transform=ax.transAxes, zorder=4))

# ── MoveNet stick-figure (standing pose) ──────────────────────────
# all coords in axes; figure centroid at (0.287, 0.640)
cx0, cy0 = .287, .660
# joint positions
joints = {
    'nose':      (cx0,     cy0),
    'l_sho':     (cx0-.044, cy0-.024),
    'r_sho':     (cx0+.044, cy0-.024),
    'l_elb':     (cx0-.060, cy0-.070),
    'r_elb':     (cx0+.060, cy0-.070),
    'l_wri':     (cx0-.062, cy0-.110),
    'r_wri':     (cx0+.062, cy0-.110),
    'l_hip':     (cx0-.030, cy0-.110),
    'r_hip':     (cx0+.030, cy0-.110),
    'l_kne':     (cx0-.038, cy0-.170),
    'r_kne':     (cx0+.038, cy0-.170),
    'l_ank':     (cx0-.042, cy0-.225),
    'r_ank':     (cx0+.042, cy0-.225),
}
bones = [('nose','l_sho'),('nose','r_sho'),
         ('l_sho','r_sho'),('l_sho','l_elb'),('r_sho','r_elb'),
         ('l_elb','l_wri'),('r_elb','r_wri'),
         ('l_sho','l_hip'),('r_sho','r_hip'),
         ('l_hip','r_hip'),
         ('l_hip','l_kne'),('r_hip','r_kne'),
         ('l_kne','l_ank'),('r_kne','r_ank')]
for a,b in bones:
    x1,y1 = joints[a]; x2,y2 = joints[b]
    ax.plot([x1,x2],[y1,y2],'-',color=CYAN,lw=2.2,
            transform=ax.transAxes,zorder=6)
for name,(jx,jy) in joints.items():
    ms = 12 if name=='nose' else 5
    ax.plot(jx,jy,'o',color=CYAN,ms=ms,
            transform=ax.transAxes,zorder=7,markeredgecolor='none')

# fall score overlay bar
ax.add_patch(FancyBboxPatch((VL,VB),VR-VL,.056,
    boxstyle="round,pad=0,rounding_size=.003",
    linewidth=0,facecolor='#000000CC',transform=ax.transAxes,zorder=8))
txt(ax, .030, .452, 'FALL SCORE', fs=6.5, color='#94a3b8', fw='700', zo=9)
txt(ax, .030, .435, '0.21', fs=16, color='white', fw='800', va='bottom', zo=9)
# bar track
ax.add_patch(FancyBboxPatch((.120,.440),.260,.012,
    boxstyle="round,pad=0,rounding_size=.002",
    linewidth=0,facecolor='#ffffff30',transform=ax.transAxes,zorder=9))
ax.add_patch(FancyBboxPatch((.120,.440),.260*.21,.012,
    boxstyle="round,pad=0,rounding_size=.002",
    linewidth=0,facecolor=SUCCESS,transform=ax.transAxes,zorder=10))
txt(ax, .405, .446, '  Safe', fs=10, color=SUCCESS, fw='800', zo=10)

# ── CLINICAL SNAPSHOT  (x:.585–.990, y:.410–.840) ─────────────────
SL,SR,SB,ST = .585,.990,.410,.840
card(ax, SL, SB, SR-SL, ST-SB)
txt(ax, .600, .820, 'Clinical Snapshot', fs=10, fw='bold')

# mini trend chart (figure coords)
ax_in = fig.add_axes([.617, .657, .218, .082])
np.random.seed(42)
sc = np.clip(np.cumsum(np.random.randn(24)*.025)+.18, .04, .60)
ax_in.plot(sc, color='#ef4444', lw=1.6)
ax_in.fill_between(range(24), sc, alpha=.12, color='#ef4444')
ax_in.axhline(.35, color=WARNING, ls='--', lw=.9)
ax_in.axhline(.65, color=DANGER,  ls='--', lw=.9)
ax_in.set_ylim(0,1); ax_in.set_xlim(0,23)
ax_in.set_xticks([]); ax_in.set_yticks([0,.35,.65,1])
ax_in.tick_params(labelsize=5.5); ax_in.set_facecolor('#f8fafc')
for sp in ax_in.spines.values(): sp.set_linewidth(.5)
txt(ax, .600, .808, 'Fall Score Trend (24 frames)', fs=6.5, color=MUTED)


# metric rows (start y=.640, step=.032, 8 rows → bottom .640-7*.032=.416 > .410 ✓)
mets = [
    ('Camera:',       'Active — /dev/video0',    SUCCESS),
    ('Mode:',         'MoveNet Lightning F16',    MUTED),
    ('Watch Loop:',   'Running (5 s interval)',   SUCCESS),
    ('Alert Queue:',  '0 pending',                MUTED),
    ('Latest Label:', 'Safe',                     SUCCESS),
    ('Fall Score:',   '0.213',                    TEXT),
    ('Motion Score:', '0.142',                    TEXT),
    ('Last Check:',   '14:32:07',                 MUTED),
]
for i,(lbl,val,col) in enumerate(mets):
    ry = .638 - i*.030
    txt(ax, .597, ry, lbl, fs=7.5, color=TEXT, fw='600')
    txt(ax, .764, ry, val, fs=7.5, color=col)

# ── BENCHMARK PANEL  (x:.010–.490, y:.228–.400) ───────────────────
BL,BR,BB,BT = .010,.490,.228,.402
card(ax, BL, BB, BR-BL, BT-BB)
txt(ax, .026, .385, 'Benchmark-Calibrated Energy', fs=9, fw='bold')
txt(ax, .026, .369, 'On-device measurement via FNB58 power meter', fs=6.5, color=MUTED)
bench = [('Latency Avg','42 ms',   TEXT,   BORDER,  '#f8fafc'),
         ('P95 Latency','58 ms',   TEXT,   BORDER,  '#f8fafc'),
         ('Throughput', '23.8/s',  ACCENT,'#dbeafe','#eff6ff'),
         ('Energy',     '2.91 mWh',SUCCESS,'#bbf7d0','#f0fdf4')]
for i,(lbl,val,col,ec,fc) in enumerate(bench):
    bx = .018+i*.118
    card(ax, bx, .244, .108, .108, fc=fc, ec=ec, shadow=False, zorder=4)
    txt(ax, bx+.054, .324, lbl, fs=6.5, color=MUTED, ha='center', fw='700', zo=5)
    txt(ax, bx+.054, .296, val, fs=11,  color=col,  ha='center', fw='800', zo=5)
txt(ax, .026, .237, 'Last run: 14:31:48  |  movenet_lightning_f16.tflite',
    fs=6.5, color=MUTED)

# ── SEVERITY REFERENCE  (x:.505–.990, y:.228–.402) ────────────────
RL2,RR2,RB2,RT2 = .505,.990,.228,.402
card(ax, RL2, RB2, RR2-RL2, RT2-RB2)
txt(ax, .521, .385, 'Severity Reference', fs=9, fw='bold')
sev = [(SUCCESS,'Safe (score < 0.35)',   'Normal posture — no action needed'),
       (WARNING, 'Warning (0.35–0.65)',  'Unusual posture — monitor closely'),
       (DANGER,  'Critical (> 0.65)',    'Fall detected — dispatch caregiver')]
for i,(col,lbl,desc) in enumerate(sev):
    ry = .358 - i*.046
    ax.plot(.522, ry, 'o', color=col, ms=8, transform=ax.transAxes, zorder=5)
    txt(ax, .538, ry,       lbl,  fs=8, color=col, fw='700')
    txt(ax, .538, ry-.018, desc, fs=7, color=MUTED)

# ── FALL EVENT TIMELINE  (x:.010–.990, y:.010–.218) ───────────────
TL,TR,TB,TT = .010,.990,.010,.222
card(ax, TL, TB, TR-TL, TT-TB)
txt(ax, .025, .206, 'Fall Event Timeline', fs=9.5, fw='bold')
txt(ax, .025, .191, 'Detections and acknowledgements — current session',
    fs=7, color=MUTED)
pill(ax, .808, .191, .072, .018, '1 alert', '#fee2e2', DANGER, fs=7)

# 3 event cards  (side-by-side, each width .318)
evts = [
    (DANGER, '14:28:14','FALL DETECTED',
     'Score=0.71, Motion=0.68 — MoveNet detected fall posture.',
     'Caregiver alert dispatched automatically.', True),
    (WARNING,'14:26:32','Warning posture',
     'Score=0.42 — Unusual lean, head_below_hips=True.',
     'Monitoring closely.', False),
    (SUCCESS,'14:23:05','Normal posture',
     'Score=0.18 — Patient standing normally.',
     '', False),
]
for i,(col,ts,title,ln1,ln2,ack) in enumerate(evts):
    ex = .015+i*.330
    card(ax, ex, .018, .314, .162,
         fc='#fff5f5' if col==DANGER else PANEL,
         ec='#fca5a5' if col==DANGER else BORDER,
         shadow=False, zorder=4)
    ax.plot(ex+.016, .165, 'o', color=col, ms=7,
            transform=ax.transAxes, zorder=5)
    txt(ax, ex+.032, .165, ts, fs=7.5, color=MUTED, zo=5)
    if ack:
        pill(ax, ex+.195, .154, .105, .020, 'Acknowledged', '#fee2e2', DANGER, fs=6)
    txt(ax, ex+.016, .143, title, fs=9, color=col, fw='bold', zo=5)
    txt(ax, ex+.016, .124, ln1,   fs=7, color=MUTED, zo=5)
    if ln2:
        txt(ax, ex+.016, .108, ln2, fs=7, color=MUTED, zo=5)
    if col==DANGER:
        btn(ax, ex+.016, .028, .155, .026, 'Acknowledge Alert',
            PANEL, DANGER, ec='#fca5a5', fs=7.5)

fig.savefig(f'{OUT}/fig16_medical_monitoring.png',
            dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.close()
print("OK: Fig 15 (medical) saved")

# ── verify paper still compiles ────────────────────────────────────
print("Both figures generated. Run pdflatex in Paper/ to verify.")
