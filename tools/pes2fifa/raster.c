/* Tiny z-buffer triangle rasteriser used for offline previews and light baking.
 *
 * verts: n*3 floats. persp=0: (x, y, key) in pixels, larger key = nearer.
 *        persp=1: (cx, cy, cw) clip coordinates, pixel = (cx/cw, cy/cw), key = 1/cw;
 *                 triangles are clipped against cw >= near.
 * tris : m*3 uint32 vertex indices.  ids: optional per-triangle id (NULL = triangle index).
 * key  : W*H floats, must be initialised by the caller (e.g. -inf).  idbuf: W*H int32 or NULL.
 * cull : 0 = none, 1 = drop triangles with negative signed area, -1 = drop positive.
 */
#include <math.h>
#include <stdint.h>
#include <stddef.h>

static void fill(const float *a, const float *b, const float *c, int32_t id, int W, int H,
                 float *key, int32_t *idbuf, int cull)
{
    float area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if (area == 0.0f || area != area) return;
    if (cull > 0 && area < 0) return;
    if (cull < 0 && area > 0) return;
    float minx = fminf(a[0], fminf(b[0], c[0])), maxx = fmaxf(a[0], fmaxf(b[0], c[0]));
    float miny = fminf(a[1], fminf(b[1], c[1])), maxy = fmaxf(a[1], fmaxf(b[1], c[1]));
    if (maxx < 0 || maxy < 0 || minx >= W || miny >= H) return;
    int x0 = (int)floorf(minx), x1 = (int)ceilf(maxx), y0 = (int)floorf(miny), y1 = (int)ceilf(maxy);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > W - 1) x1 = W - 1;
    if (y1 > H - 1) y1 = H - 1;
    float inv = 1.0f / area;
    for (int y = y0; y <= y1; y++) {
        float py = y + 0.5f;
        for (int x = x0; x <= x1; x++) {
            float px = x + 0.5f;
            float w0 = ((b[0] - px) * (c[1] - py) - (b[1] - py) * (c[0] - px)) * inv;
            float w1 = ((c[0] - px) * (a[1] - py) - (c[1] - py) * (a[0] - px)) * inv;
            float w2 = 1.0f - w0 - w1;
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;
            float k = w0 * a[2] + w1 * b[2] + w2 * c[2];
            size_t o = (size_t)y * W + x;
            if (k > key[o]) {
                key[o] = k;
                if (idbuf) idbuf[o] = id;
            }
        }
    }
}

void raster(const float *verts, const uint32_t *tris, const int32_t *ids, int64_t ntri, int W, int H,
            float *key, int32_t *idbuf, int persp, float near, int cull)
{
    for (int64_t t = 0; t < ntri; t++) {
        const float *v[3] = {verts + 3 * (size_t)tris[3 * t], verts + 3 * (size_t)tris[3 * t + 1],
                             verts + 3 * (size_t)tris[3 * t + 2]};
        int32_t id = ids ? ids[t] : (int32_t)t;
        if (!persp) {
            fill(v[0], v[1], v[2], id, W, H, key, idbuf, cull);
            continue;
        }
        /* clip against cw >= near (Sutherland-Hodgman, one plane) */
        float poly[4][3];
        int n = 0;
        for (int i = 0; i < 3; i++) {
            const float *p = v[i], *q = v[(i + 1) % 3];
            int pin = p[2] >= near, qin = q[2] >= near;
            if (pin) { poly[n][0] = p[0]; poly[n][1] = p[1]; poly[n][2] = p[2]; n++; }
            if (pin != qin) {
                float s = (near - p[2]) / (q[2] - p[2]);
                poly[n][0] = p[0] + s * (q[0] - p[0]);
                poly[n][1] = p[1] + s * (q[1] - p[1]);
                poly[n][2] = near;
                n++;
            }
        }
        if (n < 3) continue;
        float scr[4][3];
        for (int i = 0; i < n; i++) {
            float iw = 1.0f / poly[i][2];
            scr[i][0] = poly[i][0] * iw;
            scr[i][1] = poly[i][1] * iw;
            scr[i][2] = iw;
        }
        fill(scr[0], scr[1], scr[2], id, W, H, key, idbuf, cull);
        if (n == 4) fill(scr[0], scr[2], scr[3], id, W, H, key, idbuf, cull);
    }
}
