"""Procedural 3D model of the thoraco-lumbar spine (T10-L5 + sacrum, floating ribs,
intervertebral discs and a small metal clasp) for the landing-page hero.

Every bone is a signed distance field built from round cones and a kidney-shaped
body; marching cubes -> quadric decimation -> analytic normals + SDF ambient
occlusion -> one quantised binary buffer (spine-3d.bin) that the page draws with
raw WebGL. Units are millimetres, y up, z anterior.
"""
import json
import struct
import sys

import fast_simplification
import numpy as np
from skimage.measure import marching_cubes

# ---------------------------------------------------------------- primitives

def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def round_cone(p, a, b, r1, r2):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ba = b - a
    l2 = ba @ ba
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    w = pa * l2 - np.outer(y, ba)
    x2 = np.einsum("ij,ij->i", w, w)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - r1
    d1 = np.sqrt(x2 + z2) * il2 - r2
    d2 = np.sqrt(x2 + y2) * il2 - r1
    out = np.where(np.sign(z) * a2 * z2 > k, d1, np.where(np.sign(y) * a2 * y2 < k, d2, d3))
    return out


def flat_cone(p, a, b, r1, r2, scale):
    """Round cone squashed/stretched along world axes around its midpoint."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    s = np.asarray(scale, float)
    c = (a + b) / 2
    q = c + (p - c) / s
    return round_cone(q, c + (a - c) / s, c + (b - c) / s, r1, r2) * s.min()


def sphere(p, c, r):
    return np.linalg.norm(p - np.asarray(c, float), axis=1) - r


def ellipsoid(p, c, r):
    r = np.asarray(r, float)
    q = (p - np.asarray(c, float)) / r
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / r, axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)


def bone_noise(p, amp=0.07):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    n = (np.sin(1.7 * x + 0.3 * y + 1.1) * np.sin(1.9 * y + 0.2 * z) * np.sin(1.6 * z + 0.4 * x + 2.0)
         + 0.5 * np.sin(3.3 * x - 2.1 * z + 0.7) * np.sin(2.9 * y + 1.3 * x))
    return amp * n


def kidney_body(p, a, b, h, waist=0.09, flare=0.05, r=2.2, cup=0.9, bulge=0.0):
    """Vertebral body: elliptic cylinder with a concave back, waist, end-plate lips."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    hh = h / 2
    t = np.clip(y / hh, -1, 1)
    s = 1 - waist * (1 - t * t) + flare * np.exp(-((np.abs(y) - hh) / 1.8) ** 2) + bulge * (1 - t * t)
    ax, bz = a * s, b * s
    q = np.sqrt((x / ax) ** 2 + (z / bz) ** 2)
    g = np.sqrt((x / ax ** 2) ** 2 + (z / bz ** 2) ** 2)
    d2 = (q - 1) * q / np.maximum(g, 1e-6)
    R = 0.5 * a
    z0 = -b - R + 3.2
    c = np.sqrt(x * x + (z - z0) ** 2) - R
    d2 = smax(d2, -c, 3.0)
    dy = np.abs(y) - hh + cup * np.clip(1 - q * q, 0, 1)
    w0, w1 = d2 + r, dy + r
    return np.minimum(np.maximum(w0, w1), 0) + np.sqrt(np.maximum(w0, 0) ** 2 + np.maximum(w1, 0) ** 2) - r


# ---------------------------------------------------------------- bones

def vertebra(prm):
    a, b, h, kind, lt = prm["a"], prm["b"], prm["h"], prm["kind"], prm["lt"]
    hh = h / 2

    def f(p):
        d = kidney_body(p, a, b, h)
        y0 = hh * 0.22
        zP = -b - 10
        post = None
        for s in (-1, 1):
            parts = [
                # pedicle (oval, taller than wide)
                flat_cone(p, (s * 0.52 * a, y0, -b + 5), (s * 0.58 * a, y0 + 1, zP), 4.8, 4.3, (1, 1.5, 1)),
                # lamina plate towards the midline
                flat_cone(p, (s * 0.58 * a, y0, zP), (s * 1.5, y0 - 4, zP - 10), 4.0, 4.2, (1, 2.0, 1)),
            ]
            if kind == "L":
                parts += [
                    flat_cone(p, (s * 0.58 * a, y0 + 3, zP + 2), (s * (0.58 * a + lt), y0 + 6, zP - 3), 3.9, 2.9, (1, 1.45, 1)),
                    flat_cone(p, (s * (0.58 * a + 0.5), y0 + 4, zP - 2), (s * (0.58 * a + 2.5), hh + 9, zP - 5), 4.3, 3.7, (0.8, 1, 1)),
                    sphere(p, (s * (0.58 * a + 5.0), hh + 4.5, zP - 8), 3.0),
                    flat_cone(p, (s * 0.3 * a, y0 - 6, zP - 8), (s * 0.36 * a, -hh - 8, zP - 7), 3.8, 3.3, (0.85, 1, 1)),
                ]
            else:
                parts += [
                    flat_cone(p, (s * 0.58 * a, y0 + 4, zP + 1), (s * (0.58 * a + lt), y0 + 9, zP - 9), 4.4, 3.9, (1, 1.2, 1)),
                    flat_cone(p, (s * (0.55 * a), y0 + 4, zP - 1), (s * (0.55 * a + 1), hh + 8, zP - 2), 3.8, 3.2, (1, 1, 0.8)),
                    flat_cone(p, (s * 0.28 * a, y0 - 5, zP - 8), (s * 0.3 * a, -hh - 7, zP - 9), 3.6, 3.1, (1, 1, 0.8)),
                ]
            for q in parts:
                post = q if post is None else smin(post, q, 2.2)
        if kind == "L":
            sp = flat_cone(p, (0, y0 - 4, zP - 10), (0, y0 - 8, zP - 10 - 25), 3.2, 4.0, (1, 2.3, 1))
        else:
            sp = flat_cone(p, (0, y0 - 2, zP - 9), (0, y0 - 24, zP - 9 - 20), 3.0, 3.4, (1, 1.5, 1.25))
        post = smin(post, sp, 2.5)
        d = smin(d, post, 3.0)
        return d + bone_noise(p)

    ext = dict(L=(lt + 0.58 * a + 8, hh + 22, b + 8, b + 50), T=(lt + 0.58 * a + 10, hh + 34, b + 8, b + 50))[kind]
    bbox = ((-ext[0], -ext[1], -ext[3]), (ext[0], ext[1], ext[2]))
    return f, bbox


def sacrum():
    def f(p):
        d = kidney_body(p, 25, 17.5, 22, waist=0.02, flare=0.04, r=2.5, cup=0.6)
        d = smax(d, p[:, 1] - 11.0 + 0 * p[:, 0], 1.0)  # flat top endplate
        # alae (wings) sweeping laterally and down
        for s in (-1, 1):
            d = smin(d, flat_cone(p, (s * 15, 2, -4), (s * 44, -6, -12), 11.0, 8.5, (1, 1.35, 1)), 6)
            d = smin(d, ellipsoid(p, (s * 38, -18, -16), (13, 16, 9)), 7)
            # superior articular processes
            d = smin(d, flat_cone(p, (s * 16, 0, -24), (s * 18, 14, -28), 4.5, 3.8, (0.85, 1, 1)), 3)
        # tapering, backward-curving plate of fused segments
        for i in range(6):
            t = i / 5
            c = (0, -14 - 17 * i, -6 - 3.5 * i - 1.8 * i * i)
            d = smin(d, ellipsoid(p, c, (38 - 25 * t, 12 - 3 * t, 11 - 5 * t)), 6)
        # anterior sacral foramina
        for i in range(4):
            for s in (-1, 1):
                c = (s * (14 - 1.6 * i), -17 - 16 * i, 4 - 5.5 * i - 1.7 * i * i)
                d = smax(d, -sphere(p, c, 3.4 - 0.35 * i), 1.2)
        # median crest on the back
        d = smin(d, flat_cone(p, (0, -4, -26), (0, -70, -54), 3.2, 2.2, (1, 1, 1)), 3)
        return d + bone_noise(p)

    return f, ((-62, -110, -80), (62, 18, 26))


def catmull(pts, n):
    pts = np.asarray(pts, float)
    P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


def rib(ctrl, r0=3.0, r1=1.8):
    path = catmull(ctrl, 7)
    lo = path.min(0) - 8
    hi = path.max(0) + 8
    n = len(path) - 1

    def f(p):
        d = None
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            # rib head is knobby, then the shaft flattens and tapers
            ra = r0 * (1 - t0) + r1 * t0 + (1.3 if i == 0 else 0)
            rb = r0 * (1 - t1) + r1 * t1
            q = flat_cone(p, path[i], path[i + 1], ra, rb, (1, 1.35, 1))
            d = q if d is None else smin(d, q, 1.2)
        return d + bone_noise(p, 0.05)

    return f, (tuple(lo), tuple(hi))


def disc(a, b, h):
    def f(p):
        return kidney_body(p, a * 0.97, b * 0.97, h, waist=0.0, flare=0.0, r=min(2.2, h * 0.4), cup=-0.6, bulge=0.05)

    return f, ((-a - 4, -h / 2 - 3, -b - 4), (a + 4, h / 2 + 3, b + 4))


def clasp():
    """Clothing button: the foreign object the model flags (a classic DXA artefact)."""
    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        rxy = np.sqrt(x * x + y * y)
        # slightly domed disc with a raised rim, facing the viewer (+z)
        dome = z - 0.9 * (1 - (rxy / 7.5) ** 2).clip(0)
        w0, w1 = rxy - 7.5 + 0.9, np.abs(dome) - 1.0 + 0.9
        d = np.minimum(np.maximum(w0, w1), 0) + np.sqrt(np.maximum(w0, 0) ** 2 + np.maximum(w1, 0) ** 2) - 0.9
        qx = rxy - 6.6
        d = smin(d, np.sqrt(qx * qx + (z - 0.9) ** 2) - 0.9, 0.6)
        for hx, hy in ((-1.7, -1.7), (1.7, -1.7), (-1.7, 1.7), (1.7, 1.7)):
            d = smax(d, -(np.sqrt((x - hx) ** 2 + (y - hy) ** 2) - 0.8), 0.3)
        return d

    return f, ((-10, -10, -4), (10, 10, 5))


# ---------------------------------------------------------------- assembly

LEVELS = [  # bottom to top: name, a, b, h, kind, transverse length, sagittal tilt (deg), disc above
    ("L5", 25.0, 17.5, 27.0, "L", 15, 17, 10.0),
    ("L4", 24.5, 17.5, 27.0, "L", 20, 9, 10.0),
    ("L3", 24.0, 17.0, 26.0, "L", 24, 2, 9.5),
    ("L2", 23.0, 16.5, 25.0, "L", 21, -5, 9.0),
    ("L1", 22.0, 16.0, 24.0, "L", 16, -10, 7.0),
    ("T12", 20.5, 15.0, 23.0, "T", 7, -12, 6.0),
    ("T11", 19.0, 14.5, 22.0, "T", 10, -11, 5.5),
    ("T10", 17.5, 14.0, 21.0, "T", 14, -9, 0.0),
]
PART = {"S": 0, "L5": 1, "L4": 2, "L3": 3, "L2": 4, "L1": 5, "T12": 6, "T11": 7, "T10": 8, "rib": 9, "disc": 10, "metal": 11}


def rot_x(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def layout():
    """Place every piece: returns list of (name, part_id, sdf, bbox, R, t, target_tris)."""
    pieces = []
    sac_tilt = 34
    pieces.append(("S", PART["S"], *sacrum(), rot_x(sac_tilt), np.zeros(3), 3400))
    top = np.zeros(3) + rot_x(sac_tilt) @ np.array([0, 11.0, 0])  # S1 endplate centre
    prev_tilt, prev_half, gap = sac_tilt, 0.0, 10.0
    prev_a, prev_b = 25.0, 17.5
    centres = {}
    for name, a, b, h, kind, lt, tilt, disc_above in LEVELS:
        u_prev = rot_x(prev_tilt) @ np.array([0, 1.0, 0])
        u = rot_x(tilt) @ np.array([0, 1.0, 0])
        disc_c = top + u_prev * 0 + (u_prev + u) / np.linalg.norm(u_prev + u) * (gap / 2)
        da, db = (a + prev_a) / 2, (b + prev_b) / 2
        pieces.append((f"disc-{name}", PART["disc"], *disc(da, db, gap + 1.2), rot_x((tilt + prev_tilt) / 2), disc_c, 520))
        centre = disc_c + (u_prev + u) / np.linalg.norm(u_prev + u) * (gap / 2) + u * (h / 2)
        f, bb = vertebra(dict(a=a, b=b, h=h, kind=kind, lt=lt))
        pieces.append((name, PART[name], f, bb, rot_x(tilt), centre, 2500 if kind == "L" else 2300))
        centres[name] = dict(c=centre.tolist(), tilt=tilt, a=a, b=b, h=h)
        top = centre + u * (h / 2)
        prev_tilt, gap, prev_a, prev_b = tilt, disc_above, a, b

    ribs = {
        "T12": [(1, 3, -0.3), (14, 2, -8), (33, -6, -9), (53, -18, -1), (66, -32, 11)],
        "T11": [(1, 3, -0.3), (15, 2, -10), (40, -6, -13), (67, -22, -3), (83, -42, 15), (86, -58, 30)],
        "T10": [(1, 3, -0.3), (15, 2, -11), (44, -5, -15), (77, -22, -3), (96, -44, 18), (98, -60, 42), (90, -72, 62)],
    }
    for lvl, ctrl in ribs.items():
        info = centres[lvl]
        a, b = info["a"], info["b"]
        for s in (-1, 1):
            pts = [(s * (a + cx), cy, (-b * 0.3 if cz == -0.3 else -b + cz)) for cx, cy, cz in ctrl]
            f, bb = rib(pts)
            pieces.append((f"rib-{lvl}{'LR'[s > 0]}", PART["rib"], f, bb, rot_x(info["tilt"]), np.array(info["c"]), 900 if lvl != "T12" else 700))

    t12 = centres["T12"]
    metal_c = np.array(t12["c"]) + np.array([-9.0, 6.0, t12["b"] + 20.0])
    pieces.append(("clasp", PART["metal"], *clasp(), rot_x(-18), metal_c, 1600))
    return pieces, centres, metal_c


# ---------------------------------------------------------------- meshing

def mesh_piece(f, bbox, step):
    lo = np.array(bbox[0], float) - step * 2
    hi = np.array(bbox[1], float) + step * 2
    n = np.ceil((hi - lo) / step).astype(int) + 1
    xs = [lo[i] + np.arange(n[i]) * step for i in range(3)]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    vol = np.empty(len(P))
    chunk = 400000
    for i in range(0, len(P), chunk):
        vol[i:i + chunk] = f(P[i:i + chunk])
    vol = vol.reshape(n)
    verts, faces, _, _ = marching_cubes(vol, 0.0, spacing=(step, step, step))
    verts += lo
    faces = faces[:, ::-1]  # outward winding (counter-clockwise from outside)
    return verts, faces


def grad(f, p, e=0.05):
    g = np.zeros_like(p)
    for i in range(3):
        d = np.zeros(3); d[i] = e
        g[:, i] = f(p + d) - f(p - d)
    return g / np.linalg.norm(g, axis=1, keepdims=True).clip(1e-9)


def build(step=0.55):
    pieces, centres, metal_c = layout()
    world = []  # (f, R, t) for AO
    meshes = []
    for name, pid, f, bbox, R, t, target in pieces:
        v, fc = mesh_piece(f, bbox, step if pid != PART["disc"] else 0.5) if pid != PART["metal"] else mesh_piece(f, bbox, 0.22)
        if len(fc) > target:
            v, fc = fast_simplification.simplify(v.astype(np.float32), fc.astype(np.int32), target_count=target)
        # snap vertices back onto the surface and take analytic normals
        v = v.astype(float)
        for _ in range(2):
            v = v - grad(f, v) * f(v)[:, None]
        nrm = grad(f, v)
        bad = ~(np.abs(nrm).sum(1) > 0.5) | np.isnan(nrm).any(1)
        if bad.any():  # degenerate gradient (exact symmetry axis): fall back to face normals
            import trimesh
            nrm[bad] = trimesh.Trimesh(v, fc, process=False).vertex_normals[bad]
        vw = v @ R.T + t
        nw = nrm @ R.T
        meshes.append(dict(name=name, pid=pid, v=vw, n=nw, f=fc.astype(np.int64)))
        world.append((f, R, t, pid))
        print(f"{name:10s} verts={len(v):6d} tris={len(fc):6d}", file=sys.stderr)

    # recentre: spine axis (mean body centre in x/z) at the origin, height centred
    allv = np.vstack([m["v"] for m in meshes])
    body_c = np.array([c["c"] for c in centres.values()])
    off = np.array([0.0, (allv[:, 1].min() + allv[:, 1].max()) / 2, body_c[:, 2].mean()])
    for m in meshes:
        m["v"] = m["v"] - off

    def world_sdf(p):
        d = np.full(len(p), 1e9)
        for f, R, t, pid in world:
            if pid == PART["metal"]:
                continue
            q = (p + off - t) @ R  # inverse rotation
            d = np.minimum(d, f(q))
        return d

    # SDF ambient occlusion (includes neighbouring bones)
    for m in meshes:
        v, n = m["v"], m["n"]
        occ = np.zeros(len(v))
        for i in range(1, 6):
            h = 1.6 * i
            occ += (h - np.minimum(world_sdf(v + n * h), h)) / (2 ** i)
        m["ao"] = np.clip(1 - 0.55 * occ, 0, 1)

    return meshes, centres, metal_c, off


def oct_encode(n):
    n = n / np.abs(n).sum(1, keepdims=True)
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    ox = np.where(z < 0, (1 - np.abs(y)) * np.sign(x + 1e-12), x)
    oy = np.where(z < 0, (1 - np.abs(x)) * np.sign(y + 1e-12), y)
    return np.clip(np.round(np.stack([ox, oy], 1) * 127), -127, 127).astype(np.int8)


def pack(meshes, centres, metal_c, off, path, meta_path):
    order = sorted(meshes, key=lambda m: (m["pid"] == PART["disc"], m["pid"]))
    V, N, A, I, F = [], [], [], [], []
    base = 0
    ranges = {"bone": [0, 0], "disc": [0, 0]}
    for m in order:
        V.append(m["v"]); N.append(m["n"]); A.append(m["ao"]); I.append(np.full(len(m["v"]), m["pid"]))
        F.append(m["f"] + base)
        base += len(m["v"])
    V = np.vstack(V); N = np.vstack(N); A = np.concatenate(A); I = np.concatenate(I); F = np.vstack(F)
    ndisc = sum(len(m["f"]) for m in order if m["pid"] == PART["disc"])
    nbone = len(F) - ndisc
    assert len(V) < 65536, len(V)
    scale = float(np.abs(V).max()) * 1.001
    q = np.round(V / scale * 32767).astype(np.int16)
    rec = np.zeros(len(V), dtype=[("p", "<i2", 3), ("n", "i1", 2), ("ao", "u1"), ("id", "u1")])
    rec["p"] = q
    rec["n"] = oct_encode(N)
    rec["ao"] = np.round(A * 255).astype(np.uint8)
    rec["id"] = I.astype(np.uint8)
    header = struct.pack("<4sIIIf", b"SPN1", len(V), nbone * 3, ndisc * 3, scale)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(rec.tobytes())
        fh.write(F.astype("<u2").tobytes())
    meta = {k: dict(c=(np.array(v["c"]) - off).round(2).tolist(), tilt=v["tilt"], a=v["a"], b=v["b"], h=v["h"]) for k, v in centres.items()}
    meta["clasp"] = (metal_c - off).round(2).tolist()
    meta["bounds"] = [V.min(0).round(1).tolist(), V.max(0).round(1).tolist()]
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"verts={len(V)} tris={len(F)} bone={nbone} disc={ndisc} bytes={20 + rec.nbytes + F.size * 2}", file=sys.stderr)


if __name__ == "__main__":
    meshes, centres, metal_c, off = build(float(sys.argv[1]) if len(sys.argv) > 1 else 0.55)
    pack(meshes, centres, metal_c, off, "spine-3d.bin", "spine-meta.json")
