import pygame


def rgb_to_hsv(rgb:tuple[float,float,float]) -> tuple[float,float,float]:
    R,G,B = rgb
    R_prime = R/255; G_prime = G/255; B_prime = B/255
    C_max = max(R_prime, G_prime, B_prime)
    C_min = min(R_prime, G_prime, B_prime)
    delta = C_max - C_min
    if delta == 0:
        H = 0
    elif C_max == R_prime:
        H = 60 * (((G_prime - B_prime) / delta) % 6)
    elif C_max == G_prime:
        H = 60 * (((B_prime - R_prime) / delta) + 2)
    elif C_max == B_prime:
        H = 60 * (((R_prime - G_prime) / delta) + 4)
    else:
        raise RuntimeError(f"Something went wrong in the H calculation. {C_max=}, {R_prime=}, {G_prime=}, {B_prime=}")
    S = 0 if C_max == 0 else delta / C_max
    V = C_max
    return (H,S,V)

def hsv_to_rgb(hsv:tuple[float,float,float]) -> tuple[float,float,float]:
    H,S,V = hsv
    C = V * S
    X = C * (1 - abs(((H / 60) % 2) - 1))
    m = V - C
    if 0 <= H and H < 60:
        RGB_prime = (C,X,0)
    elif 60 <= H and H < 120:
        RGB_prime = (X,C,0)
    elif 120 <= H and H < 180:
        RGB_prime = (0,C,X)
    elif 180 <= H and H < 240:
        RGB_prime = (0,X,C)
    elif 240 <= H and H < 300:
        RGB_prime = (X,0,C)
    elif 300 <= H and H < 360:
        RGB_prime = (C,0,X)
    else:
        raise RuntimeError(f"Something went wrong in the (R,G,B) calculation. {H=}")
    RGB = ((rgb + m) * 255 for rgb in RGB_prime)
    return tuple(RGB)

def darken(rgb:tuple[float,float,float], amount:float) -> tuple[float,float,float]:
    h,s,v = rgb_to_hsv(rgb)
    s = min(s * 1.5, 1)
    v *= (1.0 - amount)
    return hsv_to_rgb((h,s,v))


_circle_cache = {}
def _circlepoints(r):
    r = int(round(r))
    if r in _circle_cache:
        return _circle_cache[r]
    x, y, e = r, 0, 1 - r
    _circle_cache[r] = points = []
    while x >= y:
        points.append((x, y))
        y += 1
        if e < 0:
            e += 2 * y - 1
        else:
            x -= 1
            e += 2 * (y - x) - 1
    points += [(y, x) for x, y in points if x > y]
    points += [(-x, y) for x, y in points if x]
    points += [(x, -y) for x, y in points if y]
    points.sort()
    return points

def outline_text_render(text:str, font:pygame.font.FontType, insidecolor:tuple[float,float,float], outlinecolor:tuple[float,float,float], opx:int):
    # generate a surface that's just the side of the text + 2*opx
    textsurface = font.render(text, True, insidecolor).convert_alpha()
    w = textsurface.get_width() + 2 * opx
    h = textsurface.get_height() + 2 * opx
    # make the transparent background
    osurf = pygame.Surface((w, h)).convert_alpha()
    osurf.fill((0, 0, 0, 0))
    # create a new surface we're going to place text onto
    surf = osurf.copy()
    # create the outline text on the original surface
    osurf.blit(font.render(text, True, outlinecolor).convert_alpha(), (0, 0))
    # place the outline text at various dx,dy points on the new surface
    for dx, dy in _circlepoints(opx):
        surf.blit(osurf, (dx + opx, dy + opx))
    # place the inside text on the surface
    surf.blit(textsurface, (opx, opx))
    return surf
