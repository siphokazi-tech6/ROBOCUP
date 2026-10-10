"""Athalia Mamba v102 - race-model football.

Every kick the man on the ball could make is rolled forward under the engine's
own ball physics (friction 0.985 a tick, walls returning 75%): shots at seven
points of the goal mouth, and passes or touches to himself in the 16 most
forward directions on a 10-degree ring (straight ahead out to about 75 degrees
either side) at five speeds. The keeper, anyone in our last 15 units, and
anyone with nothing playable forward look round the whole ring instead. Every
player on the pitch is raced to each ball with the engine's own movement model
(v' = 0.9 v + a, capped at 8). Our own players get a one-tick reaction delay
and half a unit less reach, so we only count passes we really collect. The
kick whose ball we win first, furthest forward and with the most time to
spare, is the one played; a ball won in the attacking third earns extra credit
for how open a first-time shot from there would be against a keeper on his
line. A shot is taken only when no opponent, the keeper included, can reach
the ball before it crosses the line.

Off the ball: the player who can reach a loose ball first, counting from when
his kick cooldown ends, goes to meet it, so a dribbler runs alongside his ball
instead of into it. When they have it, the player who can get goal-side of the
ball fastest presses it. One man always stays between the ball and our goal.
Opponents near our goal are marked, in order of danger, goal-side of where
they will be in about a second, so a runner into the box is not left behind.
The keeper narrows the angle on the line from the ball to the middle of the
goal and goes for the earliest point of any shot he can reach. On their
kickoff we stand in the passing lanes to their forwards and steer round the
centre circle, so we never give away a foul.

The search is a fixed set of candidates, so the team decides the same way on
any machine. A 12 ms clock, started when the decision starts, only trims the
widest-angle candidates on the rare tick a slow server would otherwise push
the decision past the 20 ms deadline.
"""

from math import sqrt, atan2, cos, sin
from time import perf_counter as _clock
from soccer import TeamController, TeamAction, PlayerAction

# ---------------------------------------------------------------- constants
HZ = 20.0
DT = 0.05
FR = 0.985
VMAX = 8.0
KR = 2.2
IMP = 22.0
BMAX = 30.0
HW = 50.0          # half width (x)
HH = 30.0          # half height (y)
GH = 7.0           # half goal mouth
PR = 1.0
BR = 0.45
BXM = HW - BR
BYM = HH - BR
NT = 70            # rollout horizon in ticks

# Motion: v' = 0.9 v + a, |a| <= 1 (accel * dt), |v| <= 8.
# From rest, distance covered by tick n, and how far a coasting player drifts.
R0 = [0.0]
C = [0.0]
_v = 0.0
_d = 0.0
_c = 0.0
_g = 1.0
for _n in range(1, NT + 2):
    _v = min(VMAX, 0.9 * _v + 1.0)
    _d += _v * DT
    R0.append(_d)
    _g *= 0.9
    _c += _g * DT
    C.append(_c)
# Intercept reach limits by tick, squared (as the intercept loop used to
# work them out each time).
_IRR = []
_ILIM = []
for _n in range(NT + 2):
    _rr = R0[_n] + KR
    _IRR.append(_rr * _rr)
    _lm = 0.4 * _n + KR
    _ILIM.append(_lm * _lm)
_FRP = [FR ** _n for _n in range(NT + 2)]
# Rolled distance per unit ball speed after n ticks.
SPAN = [0.0]
_s = 0.0
_g = 1.0
for _n in range(1, NT + 2):
    _s += _g * DT
    _g *= FR
    SPAN.append(_s)

HOLD = PlayerAction()

P = dict(px=0.03, pz=0.04, py=0.02, mw=0.12, tw=0.004, m0=1.0, m1=0.0, cpen=0.3, drib=0.02,
         lose=1.5, dang=3.0, dead=0.5, shot_w=10.0,
         mk_g=1.8, mk_b=0.8, mk_zone=30.0, ball_far=30.0,
         fwd_dx=18.0, fwd_y=12.0, space=0, sp_step=6.0, sp_lane=0.5, sp_x=0.1,
         sq=2.0, sqx=15.0, sqd=2.5)
LSAFE_X = P.get('lmax2', -6.0)
ODEL = int(P.get('odel', 1))
ORAD = P.get('orad', 0.5)
KO_R = 10.9
TDEL = int(P.get('tdel', 0))
TDEL_DEFAULT = TDEL

# Reach tables by tick: ours start a delay late with less reach, theirs on time.
CO = []
RO2 = []
LO2 = []
CT = []
RT2 = []
LT2 = []
for _n in range(NT + 2):
    _no = _n - ODEL if _n > ODEL else 0
    CO.append(C[_no])
    RO2.append((R0[_no] + KR - ORAD) ** 2)
    LO2.append((0.4 * _no + KR - ORAD) ** 2)
    _nt = _n - TDEL if _n > TDEL else 0
    CT.append(C[_nt])
    RT2.append((R0[_nt] + KR) ** 2)
    LT2.append((0.4 * _nt + KR) ** 2)
RS_DEL = int(P.get('rs_del', 0))
RS_W = P.get('rs_w', 3.0)
RS_DIST = P.get('rs_dist', 30.0)
CH_T = P.get('ch_t', 0.5)
ICD = int(P.get('icd', 1))
SKIP = int(P.get('skip', 1))
ISTEP = int(P.get('istep', 1))
KRC = KR * KR
PRUNE = int(P.get('prune', 1))
SQ = P['sq']
SQX = P['sqx']
KQ = 1
PUSH = 0
BEHIND = P.get('behind', 0.0)
GYS = tuple(P.get('gys', (-6.2, -4.6, -2.3, 0.0, 2.3, 4.6, 6.2)))
SPEEDS = tuple(None if s is None or s < 0 else s for s in P.get('speeds', (None, 15.0, 10.0, 8.5, 6.5)))
TWO = int(P.get('two', 0))
TOPK = int(P.get('topk', 3))
_cs = int(P.get('cstep', 20))
COARSE = [(cos(a * 0.017453292519943295), sin(a * 0.017453292519943295)) for a in range(-180, 180, _cs)]
CSP = tuple(None if v < 0 else v for v in P.get('csp', [-1, 10]))
RSP = tuple(v for v in SPEEDS if v not in CSP)
_rs = P.get('rstep', 5)
REFINE = [(cos(d * 0.017453292519943295), sin(d * 0.017453292519943295)) for d in (0.0, -_rs, _rs, -2 * _rs, 2 * _rs)]


def _mv(dx, dy, thr=1.0):
    d = sqrt(dx * dx + dy * dy)
    if d < 1e-9:
        return HOLD
    return PlayerAction(movement=(dx / d * thr, dy / d * thr))


def _strike(bvx, bvy, ux, uy, desired):
    """Kick (dir_x, dir_y, speed, power) putting the ball on ray u at about the
    desired speed (None = as fast as possible)."""
    par = bvx * ux + bvy * uy
    perp2 = bvx * bvx + bvy * bvy - par * par
    if perp2 < 0.0:
        perp2 = 0.0
    imp2 = IMP * IMP
    if perp2 >= imp2:
        return None
    span = sqrt(imp2 - perp2)
    high = par + span
    if high > BMAX - 0.3:
        high = BMAX - 0.3
    low = par - span
    if low < 0.1:
        low = 0.1
    if high <= low:
        return None
    if desired is None or desired > high:
        speed = high
    elif desired < low:
        speed = low
    else:
        speed = desired
    dx = speed * ux - bvx
    dy = speed * uy - bvy
    mag = sqrt(dx * dx + dy * dy)
    if mag < 1e-9:
        return None
    p = mag / IMP
    return (dx / mag, dy / mag, speed, 1.0 if p > 1.0 else p)


def _dirs(step):
    out = []
    a = -180
    while a < 180:
        r = a * 0.017453292519943295
        out.append((cos(r), sin(r)))
        a += step
    return out


# most forward first, so a cut-short search has looked at the useful ones
DIRS = sorted(_dirs(int(P.get('dstep', 10))), key=lambda d: -d[0])
BUDGET = P.get('budget', 0.012)


def _cands(spec):
    out = {}
    for lo, hi, step, sps in spec:
        a = lo
        while a <= hi + 1e-9:
            r = a * 0.017453292519943295
            key = round(((a + 180) % 360) - 180, 3)
            out[key] = ((cos(r), sin(r)), tuple(None if v is None or v < 0 else v for v in sps))
            a += step
    return sorted(out.values(), key=lambda c: -c[0][0])


if 'dirset' in P:
    CANDS = _cands(P['dirset'])
else:
    CANDS = [(d, SPEEDS) for d in DIRS]
ALLC = CANDS
CANDS = CANDS[:int(P.get('ndirs', 16))]
CREST = ALLC[len(CANDS):]
DEEPX = P.get('deepx', -35.0)


class MyTeam(TeamController):
    name = "Athalia_Mamba"
    version = "102"

    def __init__(self):
        self.reset(0)

    def reset(self, seed):
        self.t_act = _clock()
        self.ko_t0 = None
        self.last_t = -1
        self.tk = 0
        self.full_t = -99
        self.cache = None

    def initial_formation(self, f):
        return [(-48.0, 0.0), (-30.0, -8.0), (-30.0, 8.0), (-11.0, -14.0), (-11.0, 14.0)]

    # ------------------------------------------------------------------ act
    def act(self, obs):
        self.t_act = _clock()
        t = self.tk + 1
        self.tk = t
        ca = self.cache
        if ca is not None and t - self.full_t < SKIP and not obs.events:
            # nothing has been kicked since the last plan: keep it, unless one
            # of ours can play the ball right now
            ball = obs.ball
            bx, by = ball.position
            if bx or by:
                for p in obs.my_players:
                    if not p.kick_cooldown_ticks:
                        x, y = p.position
                        dx = bx - x
                        dy = by - y
                        if dx * dx + dy * dy <= KRC:
                            break
                else:
                    return ca
        try:
            ta = self._act(obs)
        except Exception:
            ta = TeamAction({i: HOLD for i in range(5)})
        self.full_t = t
        # the plan to repeat, with any kick taken out of it
        pl = ta.players
        rep = None
        for i, a in pl.items():
            if a.kick_direction is not None:
                if rep is None:
                    rep = dict(pl)
                rep[i] = PlayerAction(movement=a.movement)
        self.cache = ta if rep is None else TeamAction(rep)
        return ta

    def _act(self, obs):
        ta = TeamAction()
        pl = ta.players
        ball = obs.ball
        bx, by = ball.position
        bvx, bvy = ball.velocity
        mine = obs.my_players
        opp = obs.opponents
        us = [(p.position[0], p.position[1], p.velocity[0], p.velocity[1], p.kick_cooldown_ticks) for p in mine]
        th = [(p.position[0], p.position[1], p.velocity[0], p.velocity[1], p.kick_cooldown_ticks) for p in opp]
        self.us = us
        self.th = th
        kq_ = max(th, key=lambda q: q[0])
        self._kpos = (kq_[0], kq_[1])

        # ---- kickoff states: ball dead on the spot
        dead = bx == 0.0 and by == 0.0 and bvx == 0.0 and bvy == 0.0
        if dead:
            if ball.controlling_team == 0:
                return self._our_kickoff(obs, ta)
            return self._their_kickoff(obs, ta)
        self.ko_t0 = None

        # ---- ball trajectory
        traj = self._traj(bx, by, bvx, bvy)
        self.traj = traj

        # ---- intercept ticks for everyone
        ui = [self._intercept(p, traj, i == 0) for i, p in enumerate(us)]
        ti = [self._intercept(p, traj, False) for p in th]
        t_us = min(ui[1:])
        t_th = min(ti)
        chaser = 1 + ui[1:].index(t_us)
        # keeper comes for it only inside his box, and only if clearly first
        kn = min(ui[0], NT)
        kx, ky = traj[kn]
        if ui[0] + 2 < t_us and ui[0] <= t_th and kx < -38.0 and -16.0 < ky < 16.0:
            chaser = 0
            t_us = ui[0]
        winning = t_us <= t_th
        self.attack = winning or (ball.controlling_team == 0 and t_us <= t_th + 2)
        self.defending = not winning
        if not winning and chaser != 0:
            chaser = self._presser(bx, by, bvx, bvy, ui)

        done = set()
        # ---- someone of ours on the ball now?
        kicker = None
        bestd = 1e9
        for i, p in enumerate(us):
            dx = bx - p[0]
            dy = by - p[1]
            d2 = dx * dx + dy * dy
            if d2 <= KR * KR and p[4] == 0 and d2 < bestd:
                bestd = d2
                kicker = i
        if kicker is not None:
            act = self._on_ball(kicker, obs)
            if act is not None:
                pl[kicker] = act
                done.add(kicker)

        # ---- chaser
        if chaser not in done:
            if self.defending and chaser != 0:
                tx, ty = self._press_point(us[chaser], bx, by, bvx, bvy)
            else:
                n = ui[chaser]
                if n > NT:
                    n = NT
                tx, ty = traj[n]
                if BEHIND > 0.0:
                    # come onto it from behind its line, not through it
                    if n < NT:
                        dvx = traj[n + 1][0] - tx
                        dvy = traj[n + 1][1] - ty
                        dv = sqrt(dvx * dvx + dvy * dvy)
                        if dv > 0.05:
                            tx -= dvx / dv * BEHIND
                            ty -= dvy / dv * BEHIND
            pl[chaser] = self._run_to(us[chaser], tx, ty, ball_target=True)
            done.add(chaser)

        # ---- keeper
        if 0 not in done:
            pl[0] = self._keeper(obs, traj)
            done.add(0)

        # ---- off the ball
        self._shape(obs, pl, done, chaser, traj, t_us, t_th, ti)
        return ta

    # ----------------------------------------------------------- physics
    def _traj(self, x, y, vx, vy):
        self.tspd = sqrt(vx * vx + vy * vy)
        out = [(x, y)]
        app = out.append
        bym = BYM
        bxm = BXM
        gh = GH
        fr = FR
        dt = DT
        b2y = 2 * BYM
        b2x = 2 * BXM
        for n in range(NT + 1):
            if vx == 0.0 and vy == 0.0:
                # at rest (dead ball, or stopped in a goal): it stays put
                out.extend([(x, y)] * (NT + 1 - n))
                return out
            x += vx * dt
            y += vy * dt
            if y > bym:
                y = b2y - y
                vy = -0.75 * vy
            elif y < -bym:
                y = -b2y - y
                vy = -0.75 * vy
            if x > bxm:
                if -gh < y < gh:
                    x = bxm
                    vx = vy = 0.0
                else:
                    x = b2x - x
                    vx = -0.75 * vx
            elif x < -bxm:
                if -gh < y < gh:
                    x = -bxm
                    vx = vy = 0.0
                else:
                    x = -b2x - x
                    vx = -0.75 * vx
            vx *= fr
            vy *= fr
            app((x, y))
        return out

    def _intercept(self, p, traj, keeper):
        """First tick n at which the player can be within kick range of the ball."""
        px, py, vx, vy, cd = p
        r = KR
        n0 = cd if (ICD and cd) else 0
        # nobody can be in range before he could have run the gap the ball's
        # roll leaves him (bounces and the goal only shorten the ball's
        # distance from where it started): skip those ticks outright
        bx0, by0 = traj[0]
        need = sqrt((bx0 - px) ** 2 + (by0 - py) ** 2) - r - 1e-6
        if need > 0.4 * n0 + self.tspd * SPAN[n0]:
            sp = self.tspd
            if 0.4 * NT + sp * SPAN[NT] < need:
                return NT + 5
            lo = n0
            hi = NT
            while lo < hi:
                mid = (lo + hi) >> 1
                if 0.4 * mid + sp * SPAN[mid] >= need:
                    hi = mid
                else:
                    lo = mid + 1
            n0 = lo
        if ISTEP > 1:
            # coarse pass, then the exact first tick just before the hit
            n = n0
            while n <= NT:
                tx, ty = traj[n]
                c = C[n]
                dx = tx - px - vx * c
                dy = ty - py - vy * c
                rr = R0[n] + r
                if dx * dx + dy * dy <= rr * rr:
                    ex = tx - px
                    ey = ty - py
                    lim = 0.4 * n + r
                    if ex * ex + ey * ey <= lim * lim:
                        break
                n += ISTEP
            else:
                return NT + 5
            lo = n - ISTEP + 1
            if lo < n0:
                lo = n0
            n0 = lo
        IRR = _IRR
        ILIM = _ILIM
        bstep = self.tspd * DT
        n = n0
        while n <= NT:
            tx, ty = traj[n]
            ex = tx - px
            ey = ty - py
            e2 = ex * ex + ey * ey
            if e2 <= ILIM[n]:
                c = C[n]
                dx = ex - vx * c
                dy = ey - vy * c
                if dx * dx + dy * dy <= IRR[n]:
                    return n
                n += 1
            else:
                # the gap closes by at most his 0.4 plus the ball's step a
                # tick, so the next ticks that cannot close it are skipped
                k = int((sqrt(e2) - (0.4 * n + r)) / (0.4 + bstep * _FRP[n]))
                n += k if k > 1 else 1
        return NT + 5

    # ----------------------------------------------------------- pressing
    @staticmethod
    def _goalside(px, py, bx, by):
        """True when the player stands between the ball and our goal."""
        gx = -HW - bx
        gy = -by
        n = sqrt(gx * gx + gy * gy) or 1.0
        rx = px - bx
        ry = py - by
        return (rx * gx + ry * gy) / n > 0.5

    def _press_point(self, p, bx, by, bvx, bvy):
        px, py = p[0], p[1]
        dx = bx - px
        dy = by - py
        d = sqrt(dx * dx + dy * dy)
        tau = d / 14.0
        if tau > 0.6:
            tau = 0.6
        qx = bx + bvx * tau
        qy = by + bvy * tau
        gx = -HW - qx
        gy = -qy
        n = sqrt(gx * gx + gy * gy) or 1.0
        if self._goalside(px, py, bx, by):
            # straight at it, a touch goal-side
            return qx + gx / n * 0.8, qy + gy / n * 0.8
        return qx + gx / n * 3.0, qy + gy / n * 3.0

    def _presser(self, bx, by, bvx, bvy, ui):
        best = 1e9
        who = 1
        for i in range(1, 5):
            p = self.us[i]
            tx, ty = self._press_point(p, bx, by, bvx, bvy)
            dx = tx - p[0]
            dy = ty - p[1]
            t = sqrt(dx * dx + dy * dy) - (p[2] * dx + p[3] * dy) / (sqrt(dx * dx + dy * dy) + 1e-6) * 0.3
            if not self._goalside(p[0], p[1], bx, by):
                t += 4.0
            if t < best:
                best = t
                who = i
        return who

    # ------------------------------------------------------------ moving
    def _run_to(self, p, tx, ty, ball_target=False, slow=2.0):
        px, py, vx, vy, cd = p
        # aim so that velocity next tick points at target: compensate drift
        dx = tx - px - vx * 0.45 * 0.3
        dy = ty - py - vy * 0.45 * 0.3
        d = sqrt(dx * dx + dy * dy)
        if d < 1e-6:
            return HOLD
        thr = 1.0
        if not ball_target and d < slow:
            thr = d / slow
        return PlayerAction(movement=(dx / d * thr, dy / d * thr))

    # ----------------------------------------------------------- on ball
    def _reach_tables(self, kicker, extra_cd):
        """Per player (px, py, vx, vy, ready_tick, drift_x[n], drift_y[n]):
        where each one would coast to by tick n, worked out once per decision
        instead of inside every rolled kick."""
        ours = []
        for i, p in enumerate(self.us):
            ready = p[4]
            if i == kicker:
                ready = 5
            px, py, pvx, pvy = p[0], p[1], p[2], p[3]
            ours.append((px, py, pvx, pvy, ready,
                         [px + pvx * c for c in CO], [py + pvy * c for c in CO]))
        theirs = []
        for p in self.th:
            px, py, pvx, pvy = p[0], p[1], p[2], p[3]
            theirs.append((px, py, pvx, pvy, 0,
                           [px + pvx * c for c in CT], [py + pvy * c for c in CT]))
        return ours, theirs

    def _schedule(self, bx, by, ux, uy, ours, theirs):
        """For a kick along u: the first tick each player could possibly be in
        range of the ball, while it stays on that line (they are too far off
        it before then). Ours in id order, theirs in any order."""
        r = KR
        so = []
        for j, e in enumerate(ours):
            px, py, ready = e[0], e[1], e[4]
            pd = (px - bx) * uy - (py - by) * ux
            if pd < 0.0:
                pd = -pd
            m = pd - r + ORAD
            n = ODEL + int(m / 0.4) if m > 0.0 else 0
            if n < ready:
                n = ready
            so.append((n, j))
        so.sort()
        st = []
        for j, e in enumerate(theirs):
            px, py = e[0], e[1]
            pd = (px - bx) * uy - (py - by) * ux
            if pd < 0.0:
                pd = -pd
            m = pd - r
            n = TDEL + int(m / 0.4) if m > 0.0 else 0
            st.append((n, j))
        st.sort()
        return so, st

    def _rollout(self, x, y, vx, vy, ours, theirs, kicker, sched=None):
        """Roll the ball; return (goal, our_tick, our_x, our_y, our_id, their_tick, their_x, their_y)."""
        _lDT = DT
        _lBYM = BYM
        _lBXM = BXM
        _lGH = GH
        _lRO2 = RO2
        _lLO2 = LO2
        _lRT2 = RT2
        _lLT2 = LT2
        _lNT = NT
        ot = None
        tt = None
        ox = oy = tx_ = ty_ = 0.0
        oid = -1
        r = KR
        if sched is None:
            aours = list(range(len(ours)))
            athem = list(theirs)
            so = st = ()
        else:
            so, st = sched
            aours = []
            athem = []
        io = 0
        it = 0
        nso = len(so)
        nst = len(st)
        flat = True
        for n in range(1, _lNT + 1):
            x += vx * _lDT
            y += vy * _lDT
            if y > _lBYM:
                y = 2 * _lBYM - y
                vy = -0.75 * vy
                flat = False
            elif y < -_lBYM:
                y = -2 * _lBYM - y
                vy = -0.75 * vy
                flat = False
            if x > _lBXM:
                if -_lGH + 0.3 < y < _lGH - 0.3 and tt is None:
                    return (1, ot, ox, oy, oid, tt, tx_, ty_)
                x = 2 * _lBXM - x
                vx = -0.75 * vx
                flat = False
            elif x < -_lBXM:
                if -_lGH < y < _lGH and ot is None:
                    return (-1, ot, ox, oy, oid, tt, tx_, ty_)
                x = -2 * _lBXM - x
                vx = -0.75 * vx
                flat = False
            vx *= FR
            vy *= FR
            if io < nso:
                added = False
                while io < nso and (so[io][0] <= n or not flat):
                    aours.append(so[io][1])
                    io += 1
                    added = True
                if added:
                    aours.sort()
            while it < nst and (st[it][0] <= n or not flat):
                athem.append(theirs[st[it][1]])
                it += 1
            if ot is None and aours:
                ro = _lRO2[n]
                lo = _lLO2[n]
                for j in aours:
                    e = ours[j]
                    if n < e[4]:
                        continue
                    dx = x - e[5][n]
                    dy = y - e[6][n]
                    if dx * dx + dy * dy <= ro:
                        ex = x - e[0]
                        ey = y - e[1]
                        if ex * ex + ey * ey <= lo:
                            ot = n
                            ox = x
                            oy = y
                            oid = j
                            break
            if tt is None:
                rr = _lRT2[n]
                lim = _lLT2[n]
                for e in athem:
                    dx = x - e[5][n]
                    dy = y - e[6][n]
                    if dx * dx + dy * dy <= rr:
                        ex = x - e[0]
                        ey = y - e[1]
                        if ex * ex + ey * ey <= lim:
                            tt = n
                            tx_ = x
                            ty_ = y
                            break
            if ot is not None and tt is not None:
                break
            if ot is not None and n >= ot + 8:
                break
            if tt is not None and n >= tt + 8:
                break
        return (0, ot, ox, oy, oid, tt, tx_, ty_)

    @staticmethod
    def _posval(x, y):
        ay = y if y > 0 else -y
        v = P['px'] * x
        if x > 20.0:
            v += P['pz'] * (x - 20.0) - P['py'] * ay * (x - 20.0) / 30.0
        return v

    def _shotq_real(self, x, y):
        """How open a first-time shot from (x, y) would be against their
        keeper where he actually stands: the best spare reach, in units, over
        three aims (negative = he covers it)."""
        kx, ky = self._kpos
        best = -9.0
        for gy in (-5.6, 0.0, 5.6):
            dx = HW - x
            dy = gy - y
            d = sqrt(dx * dx + dy * dy) or 1.0
            ux = dx / d
            uy = dy / d
            rx = kx - x
            ry = ky - y
            a = rx * ux + ry * uy
            if a < 0.0:
                a = 0.0
            if a > d:
                a = d
            pd = rx * uy - ry * ux
            if pd < 0.0:
                pd = -pd
            n = int(a / 27.0 * HZ) + 1
            if n > NT:
                n = NT
            m = pd - R0[n] - KR
            if m > best:
                best = m
        return best

    @staticmethod
    def _shotq(x, y):
        """How open a first-time shot from (x, y) would be against a keeper
        standing on the line from there to the middle of his goal: the best
        spare reach, in units, over three aims (negative = he covers it)."""
        gx = HW - x
        d0 = sqrt(gx * gx + y * y) or 1.0
        dep = P['sqd']
        kx = HW - gx / d0 * dep
        ky = y / d0 * dep
        best = -9.0
        for gy in (-5.6, 0.0, 5.6):
            dx = HW - x
            dy = gy - y
            d = sqrt(dx * dx + dy * dy) or 1.0
            ux = dx / d
            uy = dy / d
            rx = kx - x
            ry = ky - y
            a = rx * ux + ry * uy
            if a < 0.0:
                a = 0.0
            pd = rx * uy - ry * ux
            if pd < 0.0:
                pd = -pd
            n = int(a / 27.0 * HZ) + 1
            if n > NT:
                n = NT
            m = pd - R0[n] - KR
            if m > best:
                best = m
        return best

    @staticmethod
    def _danger(x, y):
        ay = y if y > 0 else -y
        d = sqrt((x + HW) ** 2 + ay * ay)
        return P['dang'] * max(0.0, 1.0 - d / 45.0)

    def _evaldir(self, k, bx, by, bvx, bvy, ux, uy, sps, ours, theirs):
        """Best value of a kick along u over the given speeds: (value, kick)."""
        bv = -1e9
        bk = None
        sch = self._schedule(bx, by, ux, uy, ours, theirs) if PRUNE else None
        last = -1.0
        for want in sps:
            k_ = _strike(bvx, bvy, ux, uy, want)
            if k_ is None:
                continue
            sp = k_[2]
            if sp == last:
                continue
            last = sp
            res = self._rollout(bx, by, ux * sp, uy * sp, ours, theirs, k, sch)
            goal, ot, ox, oy, oid, tt, tx_, ty_ = res
            if goal == 1:
                v = 9.0
            elif goal == -1:
                v = -50.0
            elif ot is not None and (tt is None or ot < tt):
                m = (tt - ot) if tt is not None else 8
                v = self._posval(ox, oy) + P['mw'] * min(m, 8) - P['tw'] * ot
                if SQ and ox > SQX:
                    q = self._shotq_real(ox, oy) if KQ else self._shotq(ox, oy)
                    if q > 3.0:
                        q = 3.0
                    elif q < -3.0:
                        q = -3.0
                    v += SQ * q / 3.0
                need = P['m0'] + P['m1'] * ot
                if m <= need:
                    v -= P['cpen'] * (1.0 + need - m) * 0.5
                if oid == k:
                    v += P['drib']
            elif tt is not None and (ot is None or tt < ot):
                v = -P['lose'] - self._danger(tx_, ty_) + 0.01 * tx_
            else:
                # dead heat / nobody
                if ot is None:
                    x_, y_ = bx, by
                else:
                    x_, y_ = ox, oy
                v = -P['dead'] + 0.5 * self._posval(x_, y_) - 0.5 * self._danger(x_, y_)
            if v > bv:
                bv = v
                bk = k_
        return bv, bk

    def _on_ball(self, k, obs):
        bx, by = obs.ball.position
        bvx, bvy = obs.ball.velocity
        px, py = self.us[k][0], self.us[k][1]
        ours, theirs = self._reach_tables(k, 5)
        best = None
        bestv = -1e9
        t0 = self.t_act
        sc = obs.score
        chase = RS_DEL > 0 and sc[0] <= sc[1] and obs.time_remaining < CH_T
        base = self._posval(bx, by)
        # shots
        gdx = HW - bx
        if gdx < 40.0:
            for gy in GYS:
                dx = HW - bx
                dy = gy - by
                d = sqrt(dx * dx + dy * dy)
                ux = dx / d
                uy = dy / d
                k_ = _strike(bvx, bvy, ux, uy, None)
                if k_ is None or self._through_body(px, py, bx, by, ux, uy):
                    continue
                sp = k_[2]
                sch = self._schedule(bx, by, ux, uy, ours, theirs) if PRUNE else None
                res = self._rollout(bx, by, ux * sp, uy * sp, ours, theirs, k, sch)
                if res[0] == 1:
                    v = P['shot_w'] - 0.02 * d
                    if v > bestv:
                        bestv = v
                        best = (k_, ux, uy, 'shot')
                elif chase and d < RS_DIST:
                    # level or behind late on: a shot the keeper only just
                    # reaches is worth taking
                    res = self._rollout(bx, by, ux * sp, uy * sp, ours, theirs, k, sch)
                    if res[0] == 1:
                        v = RS_W - 0.02 * d
                        if v > bestv:
                            bestv = v
                            best = (k_, ux, uy, 'shot')
        # passes / touches
        if TWO:
            # coarse ring, then refine around the two best directions
            ring = []
            for (ux, uy) in COARSE:
                if self._through_body(px, py, bx, by, ux, uy):
                    continue
                v, kk = self._evaldir(k, bx, by, bvx, bvy, ux, uy, CSP, ours, theirs)
                if kk is not None:
                    ring.append((v, ux, uy, kk))
            ring.sort(key=lambda e: -e[0])
            for v, ux, uy, kk in ring[:1]:
                if v > bestv:
                    bestv = v
                    best = (kk, ux, uy, 'play')
            for v0, ux0, uy0, kk0 in ring[:TOPK]:
                for ca, sa in REFINE:
                    if _clock() - t0 > BUDGET:
                        break
                    ux = ux0 * ca - uy0 * sa
                    uy = uy0 * ca + ux0 * sa
                    if self._through_body(px, py, bx, by, ux, uy):
                        continue
                    sps = SPEEDS if (ca != 1.0 or sa != 0.0) else RSP
                    v, kk = self._evaldir(k, bx, by, bvx, bvy, ux, uy, sps, ours, theirs)
                    if kk is not None and v > bestv:
                        bestv = v
                        best = (kk, ux, uy, 'play')
        else:
            cands = CANDS
            if k == 0 or bx < DEEPX:
                # at the back every direction counts: a clearance may have
                # to go sideways, or back across a fast ball
                cands = ALLC
            for (ux, uy), sps in cands:
                if _clock() - t0 > BUDGET:
                    break
                if self._through_body(px, py, bx, by, ux, uy):
                    continue
                v, kk = self._evaldir(k, bx, by, bvx, bvy, ux, uy, sps, ours, theirs)
                if kk is not None and v > bestv:
                    bestv = v
                    best = (kk, ux, uy, 'play')
            if best is None and cands is CANDS:
                # nothing playable forward: look at the rest of the ring
                for (ux, uy), sps in CREST:
                    if _clock() - t0 > BUDGET:
                        break
                    if self._through_body(px, py, bx, by, ux, uy):
                        continue
                    v, kk = self._evaldir(k, bx, by, bvx, bvy, ux, uy, sps, ours, theirs)
                    if kk is not None and v > bestv:
                        bestv = v
                        best = (kk, ux, uy, 'play')
        if best is None:
            return None
        k_, ux, uy, kind = best
        # run with the ball's new direction
        return PlayerAction(movement=(ux, uy), kick_direction=(k_[0], k_[1]), kick_power=k_[3])

    @staticmethod
    def _through_body(px, py, bx, by, ux, uy):
        rx = px - bx
        ry = py - by
        along = rx * ux + ry * uy
        if along <= 0.0:
            return False
        perp = rx * uy - ry * ux
        return perp * perp < (PR + BR) * (PR + BR)

    # ------------------------------------------------------------ keeper
    def _keeper(self, obs, traj):
        p = self.us[0]
        px, py = p[0], p[1]
        bx, by = obs.ball.position
        bvx, bvy = obs.ball.velocity
        gx = -HW
        traj = self.traj
        # incoming shot: the rolled ball ends in our goal
        if bvx < -1.0:
            gn = None
            for n in range(1, NT + 1):
                if traj[n][0] <= -BXM + 1e-6:
                    if -GH - 0.5 < traj[n][1] < GH + 0.5:
                        gn = n
                    break
            if gn is not None:
                # earliest point on the path we can reach before it goes in
                pvx, pvy = p[2], p[3]
                for n in range(0, gn + 1):
                    tx, ty = traj[n]
                    c = C[n]
                    dx = tx - px - pvx * c
                    dy = ty - py - pvy * c
                    rr = R0[n] + 1.9
                    if dx * dx + dy * dy <= rr * rr:
                        return self._run_to(p, tx, ty, True)
                tx, ty = traj[gn]
                return self._run_to(p, tx, ty, True)
        # angle position: on the line from goal centre to ball, 2.5 units out
        dx = bx - gx
        dy = by
        d = sqrt(dx * dx + dy * dy)
        if d < 1e-6:
            d = 1e-6
        # well off the line: shots here roll, so standing out cuts the angle
        depth = 0.4 * d
        if depth < 1.5:
            depth = 1.5
        elif depth > 11.0:
            depth = 11.0
        tx = gx + dx / d * depth
        ty = dy / d * depth
        if ty > GH - 1.0:
            ty = GH - 1.0
        elif ty < -GH + 1.0:
            ty = -GH + 1.0
        return self._run_to(p, tx, ty)

    # ------------------------------------------------------------ shape
    def _shape(self, obs, pl, done, chaser, traj, t_us, t_th, ti):
        bx, by = obs.ball.position
        us = self.us
        th = self.th
        free = [i for i in range(1, 5) if i not in done]
        if not free:
            return
        targets = []
        # the last man: always between the ball and our goal
        gx = -HW - bx
        gy = -by
        gd = sqrt(gx * gx + gy * gy) or 1.0
        if self.attack:
            lx = bx - 28.0
            if lx < -36.0:
                lx = -36.0
            elif lx > -12.0:
                lx = -12.0
            # nobody of theirs up in our half to break on us: the last man
            # can come up behind the play and recycle the ball
            if lx == -12.0 and bx - 28.0 > -12.0:
                for q in th:
                    if q[0] < 0.0:
                        break
                else:
                    lx = bx - 28.0 if bx - 28.0 < LSAFE_X else LSAFE_X
            if PUSH:
                # at most one of theirs on our side of the ball: no counter to
                # fear, so the last man follows the play 20 units behind it
                kx_ = self._kpos[0]
                nup = 0
                for q in th:
                    if q[0] < bx - 2.0 and q[0] < kx_ - 0.5:
                        nup += 1
                if nup <= 1:
                    px_ = bx - 20.0
                    if px_ > 20.0:
                        px_ = 20.0
                    if px_ > lx:
                        lx = px_
            last = (lx, by * 0.25)
        else:
            k = gd * 0.45
            if k < 6.0:
                k = 6.0
            if k > gd - 4.0:
                k = max(2.5, gd - 4.0)
            last = (-HW + (-gx) / gd * k, (-gy) / gd * k)
        if self.attack:
            fx = min(bx + P['fwd_dx'], 40.0)
            fx = max(fx, 5.0)
            f1 = (fx, -P['fwd_y'])
            f2 = (fx, P['fwd_y'])
            if P['space']:
                f1 = self._space(f1, bx, by)
                f2 = self._space(f2, bx, by)
            targets = [
                last,
                f1,
                f2,
                (bx - 6.0, -by * 0.3 + (8.0 if by < 0 else -8.0)),
            ]
        else:
            carrier = min(range(5), key=lambda j: ti[j])
            marks = []
            for j in range(5):
                if j == carrier:
                    continue
                ox, oy = th[j][0], th[j][1]
                if ox > 40.0:
                    continue  # their keeper
                gx_ = -HW - ox
                gy_ = -oy
                dg = sqrt(gx_ * gx_ + gy_ * gy_) or 1.0
                bxo = bx - ox
                byo = by - oy
                db = sqrt(bxo * bxo + byo * byo) or 1.0
                mx = ox + gx_ / dg * P['mk_g'] + bxo / db * P['mk_b']
                my = oy + gy_ / dg * P['mk_g'] + byo / db * P['mk_b']
                marks.append((dg, (mx, my)))
            marks.sort()
            cover = (bx + gx / gd * 7.0, by + gy / gd * 7.0)
            near = [m for d_, m in marks if d_ < P['mk_zone']]
            far = [m for d_, m in marks if d_ >= P['mk_zone']]
            if gd > P['ball_far']:
                targets = [last] + near + [cover] + far
            else:
                targets = near + [last, cover] + far
            while len(targets) < 4:
                targets.append((-35.0, 0.0))
        # assign in priority order: each target takes the nearest free player
        used = set()
        for tx, ty in targets:
            best = None
            bd = 1e18
            for i in free:
                if i in used:
                    continue
                dx = tx - us[i][0]
                dy = ty - us[i][1]
                d2 = dx * dx + dy * dy
                if d2 < bd:
                    bd = d2
                    best = i
            if best is None:
                break
            used.add(best)
            tx = max(-HW + 2, min(HW - 2, tx))
            ty = max(-HH + 2, min(HH - 2, ty))
            pl[best] = self._run_to(us[best], tx, ty)
        for i in free:
            if i not in used:
                pl[i] = HOLD

    def _space(self, nom, bx, by):
        """Best spot near a nominal one: away from opponents, with a clear lane
        from the ball."""
        th = self.th
        best = nom
        bestv = -1e9
        step = P['sp_step']
        for ox_ in (-step, 0.0, step):
            for oy_ in (-step, 0.0, step):
                cx = nom[0] + ox_
                cy = nom[1] + oy_
                if cx > 44.0 or cy > 27.0 or cy < -27.0:
                    continue
                lx = cx - bx
                ly = cy - by
                ll = sqrt(lx * lx + ly * ly) or 1.0
                ux = lx / ll
                uy = ly / ll
                free = 12.0
                lane = 6.0
                for q in th:
                    dx = q[0] - cx
                    dy = q[1] - cy
                    d = sqrt(dx * dx + dy * dy)
                    if d < free:
                        free = d
                    rx = q[0] - bx
                    ry = q[1] - by
                    a = rx * ux + ry * uy
                    if 0.0 < a < ll:
                        pd = rx * uy - ry * ux
                        if pd < 0:
                            pd = -pd
                        if pd < lane:
                            lane = pd
                v = free + P['sp_lane'] * lane + P['sp_x'] * cx - 0.05 * (abs(ox_) + abs(oy_))
                if v > bestv:
                    bestv = v
                    best = (cx, cy)
        return best

    # ---------------------------------------------------------- kickoffs
    def _their_kickoff(self, obs, ta):
        pl = ta.players
        th = self.th
        us = self.us
        RC = KO_R
        # their outfield players, most dangerous (nearest our goal) first
        outs = sorted(range(5), key=lambda j: th[j][0] + 0.2 * abs(th[j][1]))
        outs = [j for j in outs if th[j][0] < 40.0]
        targets = []
        for j in outs[:3]:
            ox, oy = th[j][0], th[j][1]
            d = sqrt(ox * ox + oy * oy) or 1.0
            if d > RC + 3.0:
                # stand in the lane from the ball to him, a little short of him
                k = d - 2.5
                if k < RC:
                    k = RC
                targets.append((ox / d * k, oy / d * k))
            else:
                # close to the circle: goal-side of him, outside the circle
                tx, ty = ox - 2.0, oy
                dd = sqrt(tx * tx + ty * ty) or 1.0
                if dd < RC:
                    tx, ty = tx / dd * RC, ty / dd * RC
                targets.append((tx, ty))
        targets.insert(0, (-30.0, 0.0))
        pl[0] = self._run_to(us[0], -47.5, 0.0)
        free = [1, 2, 3, 4]
        used = set()
        for tx, ty in targets:
            best = None
            bd = 1e18
            for i in free:
                if i in used:
                    continue
                dx = tx - us[i][0]
                dy = ty - us[i][1]
                d2 = dx * dx + dy * dy
                if d2 < bd:
                    bd = d2
                    best = i
            if best is None:
                break
            used.add(best)
            pl[best] = self._safe_run(us[best], tx, ty)
        for i in free:
            if i not in used:
                pl[i] = self._safe_run(us[i], -20.0, 0.0)
        return ta

    def _safe_run(self, p, tx, ty):
        """Run to a spot without ever setting foot in the centre circle."""
        px, py = p[0], p[1]
        RC = KO_R
        if tx > -0.5:
            tx = -0.5
        dt = sqrt(tx * tx + ty * ty)
        if dt < RC:
            s = RC / (dt or 1.0)
            tx *= s
            ty *= s
        dx = tx - px
        dy = ty - py
        d = sqrt(dx * dx + dy * dy)
        if d < 1e-6:
            return HOLD
        ux = dx / d
        uy = dy / d
        # does the straight path cut the circle?
        along = -(px * ux + py * uy)
        if 0.0 < along < d:
            cx = px + ux * along
            cy = py + uy * along
            if cx * cx + cy * cy < RC * RC:
                # head for the tangent point on our side of the circle
                r = sqrt(px * px + py * py) or 1.0
                nx, ny = px / r, py / r
                sgn = 1.0 if (nx * uy - ny * ux) > 0 else -1.0
                ux, uy = -ny * sgn, nx * sgn
                ux += nx * 0.3
                uy += ny * 0.3
                n = sqrt(ux * ux + uy * uy)
                ux /= n
                uy /= n
        # never step inward while already on the edge
        r = sqrt(px * px + py * py)
        if r < RC + 0.6:
            nx, ny = px / (r or 1.0), py / (r or 1.0)
            inward = ux * nx + uy * ny
            if inward < 0.0:
                ux -= inward * nx
                uy -= inward * ny
                ux += nx * 0.2
                uy += ny * 0.2
                n = sqrt(ux * ux + uy * uy) or 1.0
                ux /= n
                uy /= n
        thr = 1.0 if d > 2.0 else d / 2.0
        return PlayerAction(movement=(ux * thr, uy * thr))

    def _our_kickoff(self, obs, ta):
        pl = ta.players
        t = obs.tick
        if self.ko_t0 is None:
            self.ko_t0 = t
        el = t - self.ko_t0
        spots = [(-48.0, 0.0), (-25.0, 0.0), (-6.0, 0.0), (8.0, -18.0), (8.0, 18.0)]
        for i in range(5):
            tx, ty = spots[i]
            if i == 2:
                # walk to the ball, take it near the end
                tx, ty = -1.5, 0.0
            pl[i] = self._run_to(self.us[i], tx, ty)
        if el >= 50:
            # strike from the spot using the standard evaluation
            p = self.us[2]
            dx = -p[0]
            dy = -p[1]
            if dx * dx + dy * dy <= KR * KR and p[4] == 0:
                self.traj = self._traj(0.0, 0.0, 0.0, 0.0)
                act = self._on_ball(2, obs)
                if act is not None:
                    pl[2] = act
            else:
                pl[2] = self._run_to(p, 0.0, 0.0, True)
        return ta
