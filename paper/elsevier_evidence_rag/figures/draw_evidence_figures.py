"""Rebuild the five manuscript figures as vector PDFs with Python and ReportLab."""

import math
from contextlib import contextmanager
from pathlib import Path

import reportlab
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


HERE = Path(__file__).resolve().parent
OUTPUT = HERE
WIDTH = 560
INK = "#20384A"
MUTED = "#506779"
FEATURE_LINE = "#17618B"
SECTION_FILL = "#EBF5FA"
RULE = "#94AFC1"
LINE = "#CDDCE6"
WHITE = "#FFFFFF"


class Plate:
    """Page coordinates in points, measured from the upper-left corner."""

    def __init__(self, filename, title, height):
        self.height = height
        self.pdf = Canvas(str(OUTPUT / filename), pagesize=(WIDTH, height),
                          pageCompression=1, pdfVersion=(1, 4))
        self.pdf.setTitle(title)
        self.pdf.setAuthor("GeoEvidence-RAG")
        self.pdf.setLineJoin(1)
        self.pdf.setLineCap(1)
        self.box(0, 0, WIDTH, height, WHITE)


    def path(self, points, *, edge=INK, fill=None, width=0.85, dash=(), close=False):
        c = self.pdf
        c.saveState()
        c.setLineWidth(width)
        c.setDash(list(dash))
        if edge:
            c.setStrokeColor(HexColor(edge))
        if fill:
            c.setFillColor(HexColor(fill))
        p = c.beginPath()
        p.moveTo(points[0][0], self.height - points[0][1])
        for x, y in points[1:]:
            p.lineTo(x, self.height - y)
        if close:
            p.close()
        c.drawPath(p, stroke=int(bool(edge)), fill=int(bool(fill)))
        c.restoreState()

    def box(self, x, y, w, h, fill=None, edge=None, width=0.7, dash=()):
        self.path([(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
                  fill=fill, edge=edge, width=width, dash=dash, close=True)

    def oval(self, x, y, rx, ry, edge=INK, fill=None, width=0.85):
        c = self.pdf
        c.saveState()
        c.setLineWidth(width)
        if edge:
            c.setStrokeColor(HexColor(edge))
        if fill:
            c.setFillColor(HexColor(fill))
        c.ellipse(x-rx, self.height-y-ry, x+rx, self.height-y+ry,
                  fill=int(bool(fill)), stroke=int(bool(edge)))
        c.restoreState()

    def arrow(self, *points, color=FEATURE_LINE, size=4.2, dash=(), width=0.95):
        self.path(points, edge=color, width=width, dash=dash)
        (a, b), (x, y) = points[-2:]
        angle = math.atan2(y-b, x-a)
        head = [(x-size*math.cos(angle+s), y-size*math.sin(angle+s)) for s in (-0.47, 0.47)]
        self.path([head[0], (x,y), head[1]], edge=color, width=width)

    def hatch(self, outline):
        """Hatch only solid material, leaving channels unfilled."""
        c = self.pdf
        c.saveState()
        p = c.beginPath()
        p.moveTo(outline[0][0], self.height-outline[0][1])
        for x, y in outline[1:]:
            p.lineTo(x, self.height-y)
        p.close()
        c.clipPath(p, stroke=0, fill=0)
        xs, ys = zip(*outline)
        left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        self.box(left, top, right-left, bottom-top, SECTION_FILL)
        x = left - (bottom-top)
        while x < right+1:
            self.path([(x, bottom), (x+bottom-top, top)], edge="#6F98AD", width=0.48)
            x += 4.3
        c.restoreState()
        self.path(outline, close=True, width=0.92)


    def finish(self):
        self.pdf.showPage()
        self.pdf.save()


def face(p, x, y, radius, highlight=True):
    """Sleeve-end orthographic projection. Three radii denote flange, sleeve and bore."""
    for r in (radius, radius*0.71, radius*0.44):
        p.oval(x, y, r, r, fill=SECTION_FILL if r == radius else WHITE)
    if highlight:
        p.oval(x,y,radius*0.44,radius*0.44,edge=FEATURE_LINE,width=1.65)
    for ends in [[(x-radius-4,y),(x+radius+4,y)],[(x,y-radius-4),(x,y+radius+4)]]:
        p.path(ends, edge=MUTED, dash=(5,2,0.8,2), width=0.48)


def section(p, x, y, length, radius, *, blind=False, flange=True, mark=True):
    """Longitudinal cut with an open channel or an internal end wall."""
    end = x+length
    stop = end-length*0.23 if blind else end
    bore = radius*0.44
    outer = radius*0.72 if flange else radius
    neck = x+length*0.18
    upper = [(x,y-radius), (neck,y-radius), (neck,y-outer), (end,y-outer)]
    lower = [(u,2*y-v) for u,v in upper]
    if blind:
        p.hatch(upper + list(reversed(lower)) +
                [(x,y+bore),(stop,y+bore),(stop,y-bore),(x,y-bore)])
    else:
        p.hatch(upper + [(end,y-bore),(x,y-bore)])
        p.hatch(lower + [(end,y+bore),(x,y+bore)])
    p.path([(x-4,y),(end+4,y)],edge=MUTED,width=0.45,dash=(5,2,0.8,2))
    if mark and blind:
        p.path([(stop,y-bore),(stop,y+bore)],edge=FEATURE_LINE,width=2.15)
    elif mark:
        for sign in (-1, 1):
            p.path([(end-6,y+sign*bore),(end,y+sign*bore)],edge=FEATURE_LINE,width=1.9)


# The drawing, local magnifications and paper figures share one geometry.
# Each channel has a dark line color, a tinted heading and a pale background.
ICE = ("#1F658A", "#A9D5E8", "#EDF9FE")
MINT = ("#3D6D53", "#B6D3BA", "#F0F8ED")
VIOLET = ("#73549C", "#D5C1E8", "#F8F1FD")
STEEL = ("#2C4557", "#CFDCE4", "#F4F8FA")
AMBER = ("#8A592E", "#E8CAAD", "#FFF7EA")
CORAL = ("#983C48", "#E8B2B3", "#FFF2F0")
BLACK = "#1C3042"
DRAFT = "#4C6170"
PAPER = WHITE

font_dir = Path("C:/Windows/Fonts")
for name, filename, fallback in [("AtlasText","calibri.ttf","Vera.ttf"),
                                 ("AtlasBold","calibrib.ttf","VeraBd.ttf")]:
    path = font_dir / filename
    if not path.exists():
        path = Path(reportlab.__file__).parent / "fonts" / fallback
    pdfmetrics.registerFont(TTFont(name,str(path)))


def tx(p,x,y,s,size=11,*,bold=False,col=BLACK,align="left"):
    font="AtlasBold" if bold else "AtlasText"
    wid=pdfmetrics.stringWidth(s,font,size)
    x-= {"left":0,"center":wid/2,"right":wid}[align]
    if x < -.25 or x+wid>WIDTH+.25:
        raise ValueError(f"Outside page: {s} at {x}")
    c=p.pdf
    c.setFillColor(HexColor(col))
    c.setFont(font,size)
    c.drawString(x,p.height-y-size*.31,s)


def rr(p,x,y,w,h,fill,edge=None,r=5,weight=.8,dash=()):
    c=p.pdf
    c.saveState()
    c.setLineWidth(weight)
    c.setDash(list(dash))
    if fill: c.setFillColor(HexColor(fill))
    if edge: c.setStrokeColor(HexColor(edge))
    c.roundRect(x,p.height-y-h,w,h,r,fill=int(bool(fill)),stroke=int(bool(edge)))
    c.restoreState()


def lens_rim(p, x, y, radius, theme):
    p.oval(x,y,radius+1.25,radius+1.25,edge=theme[1],width=2.0)
    p.oval(x,y,radius,radius,edge=theme[0],width=1.1)


def glyph(p,x,y,kind,col):
    def line(*points):
        p.path([(x+a,y+b) for a,b in points],edge=col,width=1.0)
    if kind=="lens":
        p.oval(x-1.4,y-1.4,4.0,4.0,edge=col,width=1.2)
        line((1.5,1.5),(6,6))
    elif kind=="layers":
        line((-6,-2),(0,-5.5),(6,-2),(0,1.5),(-6,-2))
        line((-6,1),(0,4.5),(6,1));line((-6,4),(0,7),(6,4))
    elif kind=="list":
        for dy in (-3.5,0,3.5):
            p.oval(x-5,y+dy,.7,.7,edge=None,fill=col)
            line((-2,dy),(6,dy))
    elif kind=="split":
        line((-5,0),(-1,0),(2,-4),(6,-4))
        line((-1,0),(2,4),(6,4))
    elif kind=="target":
        for r in (5.2,2.2): p.oval(x,y,r,r,edge=col)
        line((-7,0),(-4,0));line((4,0),(7,0))
    elif kind=="link":
        for dx,dy in [(-2,-2),(2,2)]: p.oval(x+dx,y+dy,3.6,2.9,edge=col,width=1.1)
    elif kind=="check":
        line((-5,-.4),(-1.6,3.4),(5.4,-4.1))
        p.oval(x,y,6.3,6.3,edge=col,width=.65)
    elif kind=="cube":
        line((-5,-3),(0,-6),(5,-3),(5,3),(0,6),(-5,3),(-5,-3),(0,0),(5,-3))
        line((0,0),(0,6))
    else:
        raise ValueError(kind)


def leader(p,points,theme):
    p.path(points,edge=theme[0],width=1.25)
    for x,y in (points[0],points[-1]):
        p.oval(x,y,1.9,1.9,edge=theme[0],fill=WHITE,width=1.05)


@contextmanager
def placed(p,x,y,scale=1):
    c=p.pdf
    c.saveState()
    c.translate(x,p.height-y-scale*p.height)
    c.scale(scale,scale)
    try: yield
    finally: c.restoreState()


def dim(p,x1,x2,y0,y,s):
    for x in (x1,x2):
        p.path([(x,y0),(x,y+2.6 if y>y0 else y-2.6)],edge=DRAFT,width=.44)
    p.arrow((x1,y),(x2,y),color=DRAFT,size=3,width=.52)
    p.arrow((x2,y),(x1,y),color=DRAFT,size=3,width=.52)
    tx(p,(x1+x2)/2,y-4.1,s,7.2,col=DRAFT,align="center")


def source_views(p):
    """Local drafting coordinates, shared with every detail window."""
    face(p,41,63,24,highlight=False)
    for x0,s in [(10,1),(72,-1)]:
        p.path([(x0,63),(x0+6*s,63)],width=.95)
        p.arrow((x0,63),(x0,54),color=INK,size=3)
        tx(p,x0,49,"A",7.4,bold=True,align="center")
    # The short leader terminates on the inner circumference.
    p.path([(48.4,55.6),(60,38),(78,38)],edge=DRAFT,width=.55)
    tx(p,70,33.2,"Ø d",7.6,col=DRAFT,align="center")
    section(p,104,63,82,24,mark=False)
    dim(p,104,186,87,96,"L")
    p.path([(104,40),(104,30),(119,30),(119,46)],edge=DRAFT,width=.48)
    p.arrow((104,30),(119,30),color=DRAFT,size=2.5,width=.45)
    p.arrow((119,30),(104,30),color=DRAFT,size=2.5,width=.45)
    tx(p,111.5,25.5,"t",7.4,col=DRAFT,align="center")
    tx(p,41,97,"SLEEVE END",7.3,bold=True,col=DRAFT,align="center")
    tx(p,150,106,"SECTION A–A",7.3,bold=True,col=DRAFT,align="center")


def sheet(p,x,y,w,*,marked=True):
    k=w/202
    with placed(p,x,y,k):
        p.box(0,0,202,132,PAPER,RULE,.7)
        p.box(3,3,196,126,None,RULE,.35)
        for xx in (49,101,153):
            p.path([(xx,0),(xx,3)],edge=RULE,width=.55)
        for yy in (32,65,99):
            p.path([(0,yy),(3,yy)],edge=RULE,width=.55)
        tx(p,8,12,"P1   /   FLANGED BUSHING",8.4,bold=True,col=DRAFT)
        tx(p,193,12,"DETAIL DRAWING",7.2,col=DRAFT,align="right")
        p.path([(6,18),(196,18)],edge=LINE,width=.5)
        source_views(p)
        if marked:
            p.box(28,50,26,26,None,ICE[0],1,(3,1.6))
            p.box(177,48.5,16,29,None,VIOLET[0],1,(3,1.6))
        p.path([(3,114),(199,114)],edge=RULE,width=.5)
        for xx in (116,166): p.path([(xx,114),(xx,129)],edge=RULE,width=.5)
        tx(p,7,122,"FEATURE   Central bore",8,col=DRAFT)
        tx(p,141,122,"Sheet 01",7.6,col=DRAFT,align="center")
        tx(p,182,122,"P1",8.3,bold=True,col=DRAFT,align="center")
    return ((x+k*41,y+k*63),(x+k*186,y+k*63))


def lens(p,cx,cy,r,source,scale,theme):
    """Magnify the same sectional/orthographic primitives used in the source sheet."""
    sx,sy=source
    c=p.pdf
    c.saveState()
    path=c.beginPath();path.circle(cx,p.height-cy,r)
    c.clipPath(path,stroke=0,fill=0)
    p.box(cx-r,cy-r,2*r,2*r,PAPER)
    with placed(p,cx-sx*scale,cy-sy*scale,scale):
        source_views(p)
    c.restoreState()
    lens_rim(p,cx,cy,r,theme)


def metal(p,x,y,k=1,*,cut=False,blind=False,theme=ICE):
    """Shaded parallel projection of a revolved, chamfered sleeve profile."""
    profile=[(0,13.5),(.55,14),(5.45,14),(6,13.45),(6,10),(35.55,10),(36,9.55)]
    profile += ([(36,0),(28,0),(28,5.9)] if blind else [(36,6.35),(35.55,5.9)])
    profile += [(.5,5.9),(0,6.4)]
    def xyz(a,r,t): return (a,r*math.cos(t),r*math.sin(t))
    def project(v):
        a,b,c=v
        return x+k*(.866*a+.5*b),y+k*(-.272*a+.471*b-.84*c)
    def depth(v):return -.42*v[0]+.727*v[1]+.544*v[2]
    def shade(f):
        base, highlight = HexColor(theme[0]).rgb(), HexColor(theme[2]).rgb()
        rgb = [round(255*(lo+(hi-lo)*f)) for lo,hi in zip(base,highlight)]
        return "#%02x%02x%02x"%tuple(rgb)
    patches=[]
    steps=38 if cut else 64
    start,span=(math.pi*.5,math.pi) if cut else (0,math.tau)
    for i,(a,r) in enumerate(profile):
        b,s=profile[(i+1)%len(profile)]
        if r==0 and s==0: continue
        for j in range(steps):
            t=start+span*j/steps
            u=start+span*(j+1)/steps
            verts=[xyz(a,r,t),xyz(b,s,t),xyz(b,s,u),xyz(a,r,u)]
            mid=(t+u)/2
            axial,radial=r-s,b-a
            norm=math.hypot(axial,radial)
            light=(-.52*axial+radial*(.17*math.cos(mid)+.80*math.sin(mid)))/norm
            color=shade(min(.98,max(.07,.54+.42*light+.045*math.cos(3*mid))))
            patches.append((sum(map(depth,verts))/4,verts,color))
    for _,verts,color in sorted(patches,key=lambda t:t[0]):
        p.path([project(v) for v in verts],edge=color,fill=color,width=.20,close=True)
    if cut:
        for t in (math.pi*.5,math.pi*1.5):
            outline=[project(xyz(a,r,t)) for a,r in profile]
            p.path(outline,edge=theme[0],fill=theme[1],width=.55,close=True)


def registry(p,x,y,values,theme,cell=30):
    for i,(name,v) in enumerate(zip(("P1","P2","P3"),values)):
        px=x+i*cell
        p.box(px,y,cell-1,11,theme[1])
        p.box(px,y+11,cell-1,13,theme[2],RULE,.35)
        tx(p,px+cell/2,y+5.5,name,9,bold=True,align="center")
        tx(p,px+cell/2,y+17.5,str(v),10.5,bold=v=="?",align="center")

def bolt_face(p,x,y,r=24,n=6,axes=True):
    """Mounting flange. H1 is the left-hand bolt hole on the cutting plane."""
    for rr_ in (r,.94*r,.44*r,.22*r):
        p.oval(x,y,rr_,rr_,edge=INK,fill=PAPER,width=.63)
    p.pdf.saveState()
    p.pdf.setDash([3,1.7,.7,1.7]);p.pdf.setStrokeColor(HexColor(DRAFT));p.pdf.setLineWidth(.42)
    p.pdf.circle(x,p.height-y,.69*r,stroke=1,fill=0);p.pdf.restoreState()
    for i in range(n):
        theta=math.pi+math.tau*i/n
        a,b=x+.69*r*math.cos(theta),y+.69*r*math.sin(theta)
        p.oval(a,b,.084*r,.084*r,edge=INK,fill=WHITE,width=.72)
        if axes:
            p.path([(a-.13*r,b),(a+.13*r,b)],edge=DRAFT,width=.28)
            p.path([(a,b-.13*r),(a,b+.13*r)],edge=DRAFT,width=.28)
    if axes:
        for pair in [[(x-r-3,y),(x+r+3,y)],[(x,y-r-3),(x,y+r+3)]]:
            p.path(pair,edge=DRAFT,width=.46,dash=(4,1.4,.6,1.4))


def bolt_section(p, x, bottom, radius):
    """Axial B–B section. Project the exact hub, bore and bolt-hole radii from bolt_face."""
    for direction in (-1, 1):
        def at(u, v):
            return x + direction * radius * u, bottom - radius * v

        p.hatch([at(.22, 0), at(.606, 0), at(.606, .43), at(.44, .43),
                 at(.44, .71), at(.22, .71)])
        p.hatch([at(.774, 0), at(1, 0), at(1, .43), at(.774, .43)])
    for offset in (-.69, 0, .69):
        cx = x + radius * offset
        p.path([(cx, bottom-radius*.81), (cx, bottom+3.5)],
               edge=DRAFT, width=.4, dash=(3, 1, .6, 1))


def flange_specimen(p, x, y, r):
    """Oblique vector rendering of the source flange, with recessed bolt holes and raised hub."""
    c = p.pdf
    squash = .57
    # y is the centre of the top mounting face; positive depth points down the page.
    def brush(clip, colors):
        c.saveState()
        c.clipPath(clip, stroke=0, fill=0)
        c.linearGradient(x-r, p.height-y+r*.5, x+r, p.height-y-r*.5,
                         [HexColor(color) for color in colors])
        c.restoreState()

    def wall(radius, top, depth):
        path = c.beginPath()
        for index in range(41):
            a = math.pi * index / 40
            pt = (x+radius*math.cos(a), p.height-top-squash*radius*math.sin(a))
            (path.moveTo if index == 0 else path.lineTo)(*pt)
        for index in reversed(range(41)):
            a = math.pi * index / 40
            path.lineTo(x+radius*math.cos(a), p.height-top-depth-squash*radius*math.sin(a))
        path.close()
        brush(path, ("#A65D6D", "#EAB3BC", "#BC7885", "#924B58"))
        c.setStrokeColor(HexColor("#9C5D6B"))
        c.setLineWidth(.48)
        c.drawPath(path, stroke=1, fill=0)

    def disk(cx, cy, radius, colors):
        circle = c.beginPath()
        circle.ellipse(cx-radius, p.height-cy-squash*radius, 2*radius, 2*radius*squash)
        brush(circle, colors)
        p.oval(cx, cy, radius, squash*radius, edge="#A96675", width=.55)

    def opening(cx, cy, radius, depth):
        aperture = c.beginPath()
        aperture.ellipse(cx-radius, p.height-cy-squash*radius, 2*radius, 2*radius*squash)
        brush(aperture, ("#593543", "#804E5C", "#B97283"))
        c.saveState()
        c.clipPath(aperture, stroke=0, fill=0)
        p.oval(cx,cy+depth,radius,radius*squash,edge="#965969",fill=PAPER,width=.5)
        c.restoreState()
        p.oval(cx,cy,radius,radius*squash,edge="#A76070",width=.55)

    c.saveState()
    bottom_depth = r*.43*math.sqrt(1-squash*squash)
    hub_height = r*.28*math.sqrt(1-squash*squash)
    wall(r,y,bottom_depth)
    silver = ("#D79CA7", "#FFF6F5", "#E9B2BB", "#F9DBDE")
    disk(x,y,r,silver)
    p.oval(x,y,.96*r,.96*r*squash,edge="#FFF7FA",width=.62)
    for i in range(6):
        angle = math.pi + math.tau*i/6
        opening(x+.69*r*math.cos(angle), y+squash*.69*r*math.sin(angle), .084*r, bottom_depth)
    wall(.44*r,y-hub_height,hub_height)
    disk(x,y-hub_height,.44*r,silver)
    opening(x,y-hub_height,.22*r,bottom_depth+hub_height)
    c.restoreState()


def flange_region(p, x, y, radius, region, scale, theme):
    """Magnify the identical drawing primitives about the centre of the marked region."""
    c = p.pdf
    clip = c.beginPath()
    clip.circle(x, p.height-y, radius)
    c.saveState()
    c.clipPath(clip, stroke=0, fill=0)
    p.box(x-radius,y-radius,2*radius,2*radius,PAPER)
    projected_radius = 31*scale
    axis = x+.69*projected_radius
    if region == "R1":
        bolt_face(p,axis,y,projected_radius)
    else:
        bolt_section(p,axis,y,projected_radius)
    c.restoreState()
    lens_rim(p,x,y,radius,theme)


def source_patch(p,x,y,w,h):
    """Small source sheet used only for the normalized-coordinate example."""
    p.box(x,y,w,h,PAPER,RULE,.65)
    bolt_face(p,x+.40*w,y+.30*h,.134*w,axes=False)
    p.path([(x+.10*w,y+.67*h),(x+.64*w,y+.67*h)],edge=RULE,width=.5)
    p.path([(x+.05*w,y+.84*h),(x+.97*w,y+.84*h)],edge=RULE,width=.55)
    p.path([(x+.7*w,y+.84*h),(x+.7*w,y+h)],edge=RULE,width=.45)


def stepped_shaft(p,x,y):
    # Sectional outlines and extension leaders for a shoulder-diameter example.
    top=[(0,-4),(3,-7),(15,-7),(15,-10),(35,-10),(35,-7),(47,-7),(48,-5)]
    outline=top+[(a,-b) for a,b in reversed(top)]
    p.path([(x+a,y+b) for a,b in outline],edge=INK,fill=SECTION_FILL,close=True,width=.75)
    p.path([(x-3,y),(x+51,y)],edge=DRAFT,width=.4,dash=(4,1,.6,1))
    p.path([(x+27,y-10),(x+36,y-17),(x+55,y-17)],edge=DRAFT,width=.5)
    tx(p,x+47,y-22,"Ø18",8.2,bold=True,align="center")


def wash(p, x, y, w, h, theme, *, strength=1):
    """Fade an accent beneath an illustration or an evidence path."""
    c = p.pdf
    c.saveState()
    c.setFillAlpha(strength)
    clip = c.beginPath()
    clip.roundRect(x, p.height - y - h, w, h, min(16, h / 2))
    c.clipPath(clip, stroke=0, fill=0)
    c.linearGradient(x, p.height-y, x+w*.82, p.height-y-h,
                     [HexColor(theme[2]), HexColor(WHITE)])
    c.restoreState()


def flow(p, points, theme, *, width=1.3, dash=()):
    p.arrow(*points, color=theme[0], size=4.6, dash=dash, width=width)


def heading(p, x, y, text, theme=STEEL, *, size=13.6):
    tx(p,x,y,text,size,bold=True,col=theme[0])


def copy_lines(p, x, y, lines, *, size=10.9, step=13, color=DRAFT, align="left"):
    for index, value in enumerate(lines):
        tx(p,x,y+index*step,value,size,col=color,align=align)


def strip(p, y, h, theme=STEEL):
    rr(p,5,y,550,h,theme[2],theme[1],r=8,weight=.6)


def dot(p, x, y, theme, size=2.6):
    p.oval(x,y,size,size,edge=theme[0],fill=WHITE,width=.95)


def pin(p, x, y, theme):
    p.oval(x,y,7,7,edge=theme[0],fill=None,width=1.25)
    p.oval(x,y,2,2,edge=WHITE,fill=theme[0],width=.65)


def coordinate_trace(p, offset):
    steps = [(11,MINT,"Crop on sheet","(.20, .10) to (.60, .50)"),
             (198,ICE,"Detection in crop","(.25, .25) to (.75, .75)"),
             (385,VIOLET,"Located on sheet","(.30, .20) to (.50, .40)")]
    for i,(x,theme,label,coords) in enumerate(steps):
        yy,w,h=248+offset,59,39
        heading(p,x,238+offset,label,theme,size=12)
        if i!=1:
            source_patch(p,x,yy,w,h)
            p.box(x+.2*w,yy+.1*h,.4*w,.4*h,None,MINT[0],.9)
            if i==2:
                p.box(x+.3*w,yy+.2*h,.2*w,.2*h,None,VIOLET[0],1.2)
        else:
            c=p.pdf
            c.saveState()
            clip=c.beginPath()
            clip.rect(x,p.height-yy-h,w,h)
            c.clipPath(clip,stroke=0,fill=0)
            source_patch(p,x-.2*w/.4,yy-.1*h/.4,w/.4,h/.4)
            c.restoreState()
            p.box(x,yy,w,h,None,MINT[0],.9)
            p.box(x+.25*w,yy+.25*h,.5*w,.5*h,None,ICE[0],1.1)
        first,second=coords.split(" to ")
        copy_lines(p,x+66,260+offset,(first,"to "+second),size=10.7,step=15,color=theme[0])
        if i != 2:
            flow(p,[(x+153,265+offset),(x+173,265+offset)],theme)


def inventory_tile(p,x,y,name,theme,*,summary=False):
    p.box(x,y,74,42,WHITE,theme[0],.6)
    p.box(x+.5,y+.5,73,11,theme[1])
    tx(p,x+4,y+6,name,9,bold=True)
    cols=("Property","Value") if summary else ("ID","Feature")
    vals=("Bores","1") if summary else ("F1","Bore")
    for dx,label,value in zip((4,42),cols,vals):
        tx(p,x+dx,y+20,label,8.8,col=DRAFT)
        tx(p,x+dx,y+34,value,9.4,bold=True)
    p.path([(x,y+26),(x+74,y+26)],edge=theme[1],width=.6)


def field(p, x, y, w, h, theme, *, phase=0):
    """Soft enclosure around a physical or documentary subject."""
    c = p.pdf
    c.saveState()
    try:
        def shape(inset):
            left, right = x+inset, x+w-inset
            top, bottom = y+inset, y+h-inset
            wx, hy = right-left, bottom-top
            turn = .09 if phase == 0 else -.09
            path = c.beginPath()
            path.moveTo(left+wx*.43, p.height-top-hy*.03)
            for a,b,d,e,f,g in [(.73,turn, .95,.03, .97,.27), (1.04,.47, .96,.53, .98,.73),
                                 (.99,.88, .83,1.02, .56,.99), (.34,1.02, .09,1.00, .05,.77),
                                 (-.03,.60, .03,.42, .035,.27), (.045,.10, .19,.02, .43,.03)]:
                path.curveTo(left+a*wx,p.height-top-b*hy,left+d*wx,p.height-top-e*hy,
                             left+f*wx,p.height-top-g*hy)
            path.close()
            return path
        outline = shape(0)
        c.setLineWidth(.65)
        c.setStrokeColor(HexColor(theme[1]))
        c.drawPath(outline,fill=0,stroke=1)
        c.clipPath(outline,fill=0,stroke=0)
        c.linearGradient(x,p.height-y,x+w,p.height-y-h,
                         [HexColor(theme[1]),HexColor(theme[2]),HexColor(WHITE)], [0,.5,1])
        c.setStrokeColor(HexColor(WHITE))
        c.setLineWidth(1.2)
        c.drawPath(shape(3),fill=0,stroke=1)
    finally:
        c.restoreState()


def route(p, start, control1, control2, end, theme, *, bold=False, partial=False):
    """Smooth transition with a fine highlight along the contested paths."""
    c = p.pdf
    thickness = 3.0 if bold else 1.15
    head = 6 if bold else 4.4
    vx,vy = end[0] - control2[0],end[1] - control2[1]
    length = math.hypot(vx,vy)
    ux,uy = vx/length,vy/length
    ex,ey = end[0]-head*.72*ux,end[1]-head*.72*uy
    trajectory = c.beginPath()
    trajectory.moveTo(start[0],p.height-start[1])
    trajectory.curveTo(control1[0],p.height-control1[1],control2[0],p.height-control2[1],ex,p.height-ey)
    c.saveState()
    c.setStrokeColor(HexColor(theme[1] if bold else theme[0]))
    c.setLineWidth(thickness)
    if partial:
        c.setDash([3,2])
    c.drawPath(trajectory,stroke=1,fill=0)
    if bold:
        c.setStrokeColor(HexColor(theme[0]))
        c.setLineWidth(.8)
        c.drawPath(trajectory,stroke=1,fill=0)
    c.restoreState()
    p.path([end,(end[0]-head*ux+head*.46*uy,end[1]-head*uy-head*.46*ux),
                 (end[0]-head*ux-head*.46*uy,end[1]-head*uy+head*.46*ux)],
           edge=None,fill=theme[0],close=True)


def accent_chip(p,x,y,s,theme,*,size=9.8):
    w = pdfmetrics.stringWidth(s,"AtlasBold",size)+13
    rr(p,x-w/2,y-7.4,w,14.8,theme[0],r=7,weight=0)
    tx(p,x,y+.1,s,size,bold=True,col=WHITE,align="center")


def status_mark(p,x,y,theme,*,ok=True):
    p.oval(x,y,5.0,5.0,edge=theme[0],fill=theme[2],width=.85)
    if ok:
        p.path([(x-2.6,y),(x-.7,y+1.9),(x+2.5,y-2)],edge=theme[0],width=1)
    else:
        p.path([(x-2,y),(x+2,y)],edge=theme[0],width=1.1)


def shadow(p,x,y,rx,ry,theme):
    for factor,col in [(1,theme[2]),(.72,theme[1])]:
        p.oval(x,y,rx*factor,ry*factor,edge=None,fill=col)


def aperture(p,x,y,theme,*,vertical=False):
    """Projected hole opening and its datum crosshair."""
    p.oval(x,y,10,10,edge=theme[1],fill=theme[2],width=1.2)
    p.oval(x,y,6.8,5.0 if vertical else 6.8,edge=theme[0],fill=WHITE,width=1.25)
    for a,b in [((x-12,y),(x+12,y)),((x,y-12),(x,y+12))]:
        p.path([a,b],edge=theme[0],width=.4,dash=(3,1.2,.6,1.2))


def slab(p,x,y,w,h,theme,rows):
    """An engineering record with a visible backing layer and traceable value cells."""
    rr(p,x+4,y+4,w,h,theme[1],r=6,weight=0)
    rr(p,x,y,w,h,WHITE,theme[1],r=6,weight=.85)
    for i,(name,value) in enumerate(rows):
        yy=y+13+i*15
        if i:
            p.path([(x+7,yy-7),(x+w-7,yy-7)],edge=theme[1],width=.45)
        tx(p,x+8,yy,name,10.6,bold=True,col=theme[0])
        tx(p,x+w-8,yy,value,10.5,align="right")


def part_shadow(p,x,y,k,theme,*,cut=False,blind=False):
    shadow(p,x+14*k,y+5.4*k,16*k,2.2*k,theme)
    metal(p,x,y,k,cut=cut,blind=blind,theme=theme)


def workflow():
    p = Plate("workflow-plate.pdf", "GeoEvidence-RAG: an evidence chain for a physical feature", 369)
    heading(p,9,14,"Is H1 a through-hole?",STEEL,size=20)
    tx(p,549,14,"GEOEVIDENCE-RAG",10.8,bold=True,col=ICE[0],align="right")
    field(p,4,31,211,226,CORAL)
    field(p,337,34,216,222,ICE,phase=1)
    heading(p,20,53,"A   Physical feature",CORAL,size=15.2)
    heading(p,357,53,"Claim-level evidence",ICE,size=15.5)
    tx(p,20,70,"Mounting flange · sheet F/01",10.5,bold=True,col=CORAL[0])
    shadow(p,94,155,69,12,CORAL)
    flange_specimen(p,94,108,57)
    px,py=94-.69*57,108
    pin(p,px,py,CORAL)
    leader(p,[(px,py),(37,86),(21,86)],CORAL)
    tx(p,11,78,"H1",11.8,bold=True,col=CORAL[0])

    rr(p,22,174,167,66,WHITE,CORAL[1],r=4,weight=.7)
    bolt_face(p,65,199,19)
    bolt_section(p,140,215,27)
    dot(p,52,199,ICE,2.5)
    dot(p,121.4,215,VIOLET,2.5)
    tx(p,66,228,"End view",10.1,bold=True,align="center")
    tx(p,139,228,"Section B–B",10.1,bold=True,align="center")
    p.path([(52,199),(39,191),(39,167),(166,167)],edge=ICE[0],width=.75)
    route(p,(166,167),(195,166),(176,88),(218,85),ICE,bold=True)
    p.path([(121.4,215),(121.4,219),(169,219)],edge=VIOLET[0],width=.75)
    route(p,(169,219),(198,218),(192,180),(216,175),VIOLET,bold=True)
    flange_region(p,245,86,22,"R1",2.45,ICE)
    flange_region(p,244,172,22,"R2",2.45,VIOLET)
    tx(p,245,117,"R1 · bolt-hole rim",10.8,bold=True,col=ICE[0],align="center")
    tx(p,245,203,"R2 · open outlet",10.8,bold=True,col=VIOLET[0],align="center")
    route(p,(270,86),(294,86),(293,114),(320,120),ICE,bold=True)
    route(p,(268,171),(293,165),(298,147),(322,136),VIOLET,bold=True)
    aperture(p,334,129,ICE)
    route(p,(347,129),(355,129),(359,129),(368,129),ICE,bold=True)
    slab(p,374,79,158,78,ICE,[("Feature key","F : H1"),("Identity","End view [R1]"),
                               ("Termination","Section [R2]"),("Provenance","Sheet · view · box")])
    route(p,(452,163),(452,174),(452,176),(452,186),MINT)
    rr(p,364,189,169,43,MINT[2],MINT[1],r=7,weight=.7)
    status_mark(p,375,207,MINT)
    tx(p,386,204,"H1 passes through",12.8,bold=True,col=MINT[0])
    tx(p,386,220,"the flange.   [R1, R2]",11.6,bold=True,col=MINT[0])

    heading(p,8,266,"B   From retrieval to support",ICE,size=13.7)
    strip(p,278,45,ICE)
    for x, title, desc, color in [(17,"Global CLIP","Find candidate sheets",CORAL),
                                  (153,"Local MLP","Compare outlet vs. wall",VIOLET),
                                  (292,"Evidence selector","Keep identity and type",ICE),
                                  (429,"Collection scope",None,MINT)]:
        heading(p,x,290,title,color,size=12.1)
        if desc:
            tx(p,x,307,desc,10.5,col=DRAFT)
    registry(p,431,298,(12,18,24),MINT,31)
    tx(p,541,316,"cm³",9.4,col=DRAFT,align="center")
    for x,y in [(123,301),(260,301)]:
        route(p,(x,y),(x+4,y),(x+9,y),(x+17,y),ICE)
    p.path([(416,285),(416,317)],edge=ICE[1],width=.75)
    heading(p,8,341,"Offline training",VIOLET,size=12.1)
    for x,title,desc in [(100,"Hard pairs","One feature edited"),(239,"Frozen CLIP → small MLP","Train the reranker"),
                         (420,"Family-level split","Keep views and variants together")]:
        tx(p,x,340,title,11.1,bold=True)
        tx(p,x,355,desc,10.3,col=DRAFT)
    p.finish()


def counterexamples():
    p = Plate("counterexample-plate.pdf", "Comparing retrieval and geometric evidence", 374)
    heading(p,7,14,"From drawing retrieval to verified evidence",STEEL,size=21)

    # Both panels use the same source records and the same three comparisons.
    rr(p,5,35,98,301,VIOLET[2],VIOLET[1],r=9,weight=.75)
    heading(p,13,48,"Shared archive",VIOLET,size=12.7)
    for y,part,volume,theme,scale in [(64,"P1",12,ICE,1.00),
                                      (123,"P2",18,CORAL,1.00),
                                      (182,"P3",24,MINT,1.04)]:
        rr(p,12,y,84,52,WHITE,theme[1],r=6,weight=.65)
        part_shadow(p,23,y+31,scale,theme,blind=part=="P2")
        tx(p,89,y+18,part,11.8,bold=True,col=theme[0],align="right")
        tx(p,89,y+34,f"{volume} cm³",10.2,col=DRAFT,align="right")

    for y,icon,label in [(248,"layers","Drawing views"),(286,"list","Part registry")]:
        rr(p,12,y,84,31,WHITE,VIOLET[1],r=5,weight=.6)
        glyph(p,24,y+15,icon,VIOLET[0])
        tx(p,35,y+15,label,9.6,bold=True,col=VIOLET[0])

    columns = (118,264,410)
    for x,title,detail,icon,theme in zip(
        columns,
        ("Step 1 · Retrieve", "Step 2 · Match features", "Step 3 · Verify scope"),
        ("Hole termination", "Cross-view identity", "Maximum volume"),
        ("lens", "link", "list"),
        (ICE,VIOLET,MINT),
    ):
        rr(p,x,35,137,33,theme[2],theme[1],r=6,weight=.7)
        glyph(p,x+13,50,icon,theme[0])
        heading(p,x+25,45,title,theme,size=11)
        tx(p,x+25,58,detail,9.2,col=DRAFT)

    for y,h,letter,name,context,theme in [
        (76,124,"A","Conventional whole-drawing retrieval","Fixed top-K shortlist",CORAL),
        (210,126,"B","GeoEvidence-RAG","Local features + registry",ICE),
    ]:
        rr(p,112,y,442,h,theme[2],theme[1],r=9,weight=.9)
        rr(p,119,y+6,20,20,WHITE,theme[1],r=5,weight=.75)
        tx(p,129,y+16,letter,14.2,bold=True,col=theme[0],align="center")
        name_x = 146
        if letter == "B":
            rr(p,145,y+6,42,20,ICE[0],r=5,weight=0)
            tx(p,166,y+16,"Ours",13.3,bold=True,col=WHITE,align="center")
            name_x = 195
        heading(p,name_x,y+16,name,theme,size=14.3)
        tx(p,546,y+16,context,10.3,col=theme[0],align="right")
        p.arrow((103,y+16),(111,y+16),color=theme[0],size=3.7,width=.9)

    p.path([(111,205),(553,205)],edge="#C64951",width=1.35,dash=(5,3))

    for y,titles,theme in [
        (106,("Global visual match","Independent regions","Top-K shortlist"),CORAL),
        (240,("Local section evidence","Cross-view feature link","Registry audit"),ICE),
    ]:
        for x,title in zip(columns,titles):
            rr(p,x,y,137,87,WHITE,theme[1],r=6,weight=.7)
            heading(p,x+8,y+11,title,theme,size=11.2)
            p.path([(x+8,y+20),(x+129,y+20)],edge=theme[1],width=.5)

    # A: retrieved images can look relevant while leaving decisive checks open.
    face(p,151,147,12,highlight=False)
    face(p,221,147,12,highlight=False)
    p.path([(168,147),(204,147)],edge=CORAL[0],dash=(2.5,2),width=.9)
    tx(p,186,140,"Similar",8.8,bold=True,col=CORAL[0],align="center")
    face(p,291,147,11)
    section(p,321,147,64,11,mark=False)
    p.path([(306,147),(315,147)],edge=CORAL[0],dash=(2,2),width=.9)
    for x,label in [(151,"P1 end view"),(221,"P2 end view"),
                    (291,"End view"),(353,"Axial section")]:
        tx(p,x,165,label,9.1,col=DRAFT,align="center")
    registry(p,423,130,(12,18,"–"),CORAL,37)
    p.oval(478.5,147.5,13,6,edge=CORAL[0],width=1.15)
    tx(p,478.5,165,"P3 omitted from shortlist",9.5,col=CORAL[0],align="center")

    # B: the local observations, feature key and registry establish distinct claims.
    section(p,129,276,49,11)
    section(p,192,276,51,11,blind=True)
    dot(p,178,276-11*.44,ICE,1.9)
    dot(p,243-51*.23,276,CORAL,1.9)
    tx(p,153,297,"Open outlet",9.2,bold=True,col=ICE[0],align="center")
    tx(p,218,297,"End wall",9.2,bold=True,col=CORAL[0],align="center")
    face(p,289,273,10)
    section(p,322,273,63,10)
    route(p,(289,285),(292,291),(300,296),(309,297),ICE)
    route(p,(369,284),(366,291),(363,294),(355,297),VIOLET)
    accent_chip(p,332,296,"P1 : F1",VIOLET,size=9.5)
    registry(p,423,263,(12,18,24),MINT,37)
    p.oval(515.5,280.5,13,6,edge=MINT[0],width=1.15)
    tx(p,478.5,298,"Volumes available: 3 / 3",9.5,col=MINT[0],align="center")

    for y,outcomes,theme,ok in [
        (175,("Termination unverified","Feature ID unverified","P2 leads shortlist"),CORAL,False),
        (309,("P1 through · P2 blind","One bore counted","P3 largest · 24 cm³"),ICE,True),
    ]:
        for x,outcome in zip(columns,outcomes):
            rr(p,x+5,y,127,15,theme[2],r=4,weight=0)
            status_mark(p,x+14,y+7.5,theme,ok=ok)
            tx(p,x+23,y+7.5,outcome,10.3,bold=True,col=theme[0])

    strip(p,344,24,AMBER)
    tx(p,14,356,"If P3 has no volume",11,bold=True,col=AMBER[0])
    p.arrow((119,356),(136,356),color=AMBER[0],size=4,width=.9)
    tx(p,143,356,"18 cm³ is the largest observed value",10.8)
    p.arrow((315,356),(333,356),color=AMBER[0],size=4,width=.9)
    tx(p,341,356,"Collection maximum unresolved",11,bold=True,col=AMBER[0])
    p.finish()


def study_map():
    p = Plate("study-plate.pdf", "Six method stages around a physical feature", 422)
    heading(p,9,14,"From physical geometry to a verified claim",STEEL,size=19.9)
    field(p,5,39,232,256,CORAL)
    field(p,238,39,317,256,ICE,phase=1)
    heading(p,16,55,"3.1  Plan the query",CORAL,size=13.9)
    copy_lines(p,16,75,("Local · does the bore pass through?", "Collection · which part is largest?"),
               size=11.4,step=15)
    heading(p,253,55,"3.3  Rerank using local geometry",ICE,size=14)
    copy_lines(p,253,75,("Similar exterior · check the bore termination",),size=11.2)
    part_shadow(p,41,168,3.65,CORAL)
    part_shadow(p,271,179,4.03,ICE,cut=True)
    route(p,(172,165),(194,142),(213,131),(239,142),VIOLET,bold=True)
    accent_chip(p,211,126,"Axial cut",VIOLET,size=10.3)
    tx(p,83,217,"Flanged bushing P1",12,bold=True,col=CORAL[0])
    heading(p,17,238,"3.2  Link local evidence",VIOLET,size=13.2)
    face(p,40,262,12)
    section(p,72,262,61,13)
    route(p,(138,262),(147,262),(148,262),(156,262),VIOLET)
    tx(p,161,262,"P1 : F1",12.7,bold=True,col=VIOLET[0])

    leader(p,[(271,178),(249,182),(235,182)],ICE)
    tx(p,233,195,"Entry",10.7,bold=True,col=ICE[0])
    leader(p,[(396.6,139.5),(415,99),(450,99)],ICE)
    tx(p,453,98,"Outlet",10.7,bold=True,col=ICE[0])
    tx(p,332,234,"Open channel",12,bold=True,col=ICE[0],align="center")
    section(p,255,262,75,11)
    section(p,343,262,70,11,blind=True)
    tx(p,292,278,"Through",10.2,bold=True,col=ICE[0],align="center")
    tx(p,376,278,"Blind",10.2,bold=True,col=CORAL[0],align="center")

    rr(p,426,110,111,143,MINT[2],MINT[1],r=10,weight=.75)
    heading(p,433,125,"3.5  Scope audit",MINT,size=12.9)
    registry(p,434,149,(12,18,"?"),MINT,31)
    tx(p,434,187,"P3 has no volume.",10.8,bold=True,col=AMBER[0])
    copy_lines(p,434,205,("Known max: 18 cm³", "Archive max:", "unresolved"),size=10.7)

    route(p,(183,275),(180,286),(158,293),(155,303),VIOLET)
    route(p,(323,288),(309,300),(224,299),(193,306),ICE)
    route(p,(486,258),(485,278),(459,292),(452,305),MINT)
    strip(p,305,53,ICE)
    heading(p,14,319,"3.4  Select under budget",ICE,size=13.1)
    tx(p,14,337,"Count: inventory  ·  Type: axial section",11,col=DRAFT)
    route(p,(239,335),(245,335),(253,335),(266,335),ICE,bold=True)
    heading(p,282,319,"3.6  Verify and qualify",MINT,size=13.1)
    copy_lines(p,282,334,("Cite the section for type.","Qualify claims when a required value is missing."),
               size=10.7,step=13)

    for x,title,desc,theme in [(7,"Dimension","Shoulder Ø18 mm",CORAL),(193,"Count","6 bolt-circle holes",ICE),
                               (379,"Termination","Blind · end wall",VIOLET)]:
        wash(p,x,369,173,45,theme)
        heading(p,x+59,381,title,theme,size=11.8)
        tx(p,x+59,402,desc,10.9,bold=True)
    stepped_shaft(p,9,397)
    bolt_face(p,219,392,15)
    section(p,383,393,48,10,blind=True)
    p.finish()


def evidence_unit():
    p = Plate("feature-record-plate.pdf", "Geometric evidence across two aligned projections", 347)
    heading(p,9,14,"Two views, one physical feature",STEEL,size=20)
    field(p,5,31,229,216,CORAL)
    field(p,345,32,209,213,ICE,phase=1)
    heading(p,14,46,"A   Document and local geometry",CORAL,size=13.5)
    heading(p,358,46,"B   Local geometry record",ICE,size=14.5)
    part_shadow(p,27,82,1.36,CORAL)
    tx(p,94,78,"Flanged bushing P1",11.4,bold=True,col=CORAL[0])
    tx(p,94,92,"Section A–A crosses the bore axis",9.7,col=DRAFT)
    p.box(21,116,201,131,CORAL[1])
    sheet(p,16,111,201)
    lens(p,276,103,28,(41,63),2.2,ICE)
    lens(p,276,190,28,(186,63),2.0,VIOLET)
    leader(p,[(57,163),(78,128),(232,128),(250,117)],ICE)
    leader(p,[(201,174),(222,174),(247,185)],VIOLET)
    tx(p,276,140,"R1 · bore edge",11,bold=True,col=ICE[0],align="center")
    tx(p,276,226,"R2 · open outlet",11,bold=True,col=VIOLET[0],align="center")
    route(p,(306,103),(320,103),(324,124),(343,126),ICE,bold=True)
    route(p,(307,186),(322,185),(328,159),(343,149),VIOLET,bold=True)
    aperture(p,350,137,ICE)
    route(p,(362,137),(371,137),(368,137),(377,137),ICE)
    heading(p,372,76,"P1 : F1",VIOLET,size=18.0)
    tx(p,541,76,"Central bore",11.6,bold=True,align="right")
    slab(p,381,92,158,84,ICE,[("View regions","R1 + R2"),("Termination","Through"),
                               ("Source","Sheet · view · box"),("CAD faces","When supplied")])
    rr(p,369,198,168,29,MINT[2],MINT[1],r=7,weight=.6)
    status_mark(p,381,212,MINT)
    tx(p,395,212,"One identity for both views",11.3,bold=True,col=MINT[0])
    strip(p,257,82,VIOLET)
    heading(p,12,270,"C   Map the detected region back to the original sheet",VIOLET,size=13.5)
    coordinate_trace(p,49)
    p.finish()


def budget_trace():
    p = Plate("selection-plate.pdf", "Retain complementary evidence within the context budget", 350)
    heading(p,9,14,"How many bores, and what type?",STEEL,size=20)
    field(p,4,38,174,205,CORAL)
    field(p,355,38,198,205,ICE,phase=1)
    heading(p,16,54,"Source observations",CORAL,size=14.5)
    rr(p,21,68,74,42,CORAL[1],r=4,weight=0)
    inventory_tile(p,16,64,"Inventory A",CORAL)
    rr(p,26,125,74,42,VIOLET[1],r=4,weight=0)
    inventory_tile(p,21,121,"Inventory B",VIOLET,summary=True)
    section(p,16,198,76,17)
    dot(p,92,191,ICE,2.4)
    tx(p,53,226,"Section A–A",10.9,bold=True,align="center")
    copy_lines(p,104,79,("One keyed", "bore · F1"),size=11.3)
    copy_lines(p,104,139,("Same count", "restated"),size=11.3)
    copy_lines(p,104,195,("Open outlet", "visible"),size=11.3)
    for yy,color in [(91,CORAL),(149,VIOLET),(198,ICE)]:
        route(p,(167,yy),(181,yy),(181,yy),(191,yy),color,bold=True)

    heading(p,194,54,"A   Condition coverage",ICE,size=13.7)
    tx(p,263,71,"Count",11.8,bold=True,col=CORAL[0],align="center")
    tx(p,319,71,"Type",11.8,bold=True,col=ICE[0],align="center")
    for yy,label,a,b in [(91,"A",1,0),(149,"B",1,0),(198,"Section",0,1)]:
        tx(p,198,yy,label,11.8,bold=True)
        for xx,v,color in [(263,a,CORAL),(319,b,ICE)]:
            p.oval(xx,yy,11,11,edge=color[1],fill=color[2] if v else WHITE,width=1.1)
            tx(p,xx,yy,str(v),12.1,bold=bool(v),col=color[0] if v else DRAFT,align="center")
        p.path([(196,yy+18),(339,yy+18)],edge=LINE,width=.58)
    tx(p,195,228,"Equal weights · unit record cost",10.6,col=DRAFT)

    heading(p,365,54,"B   After selecting A",ICE,size=14.3)
    status_mark(p,374,87,CORAL)
    tx(p,386,85,"Count already covered",11.9,bold=True)
    tx(p,386,102,"First tie resolved by input order",10.7,col=DRAFT)
    route(p,(334,149),(353,149),(356,144),(367,144),VIOLET,partial=True)
    status_mark(p,378,144,VIOLET,ok=False)
    tx(p,390,141,"Skip duplicate inventory B",11.8,bold=True,col=VIOLET[0])
    tx(p,390,158,"Gain 0 · utility ≤ 0",11.3,col=DRAFT)
    route(p,(333,198),(351,198),(359,194),(367,194),ICE,bold=True)
    status_mark(p,378,194,MINT)
    tx(p,390,191,"Add complementary section",11.8,bold=True,col=MINT[0])
    tx(p,390,209,"Gain 1 · utility ≥ 0.8",11.4,col=MINT[0])

    strip(p,255,86,STEEL)
    heading(p,12,269,"C   What the reader can substantiate",STEEL,size=13.8)
    wash(p,11,279,246,56,AMBER)
    wash(p,290,279,257,56,MINT)
    accent_chip(p,36,291,"B = 1",AMBER)
    accent_chip(p,312,291,"B = 2",MINT)
    face(p,44,317,12,highlight=False)
    section(p,296,320,60,9)
    tx(p,74,292,"One bore",13.3,bold=True)
    tx(p,74,313,"Type unresolved",11.5,col=AMBER[0])
    route(p,(245,313),(257,313),(270,313),(282,313),ICE,bold=True)
    tx(p,368,292,"One through bore",13.3,bold=True,col=MINT[0])
    tx(p,368,314,"Cite inventory A and section A–A",10.9,col=MINT[0])
    p.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    OUTPUT = parser.parse_args().output_dir
    OUTPUT.mkdir(parents=True, exist_ok=True)
    workflow()
    counterexamples()
    study_map()
    evidence_unit()
    budget_trace()
