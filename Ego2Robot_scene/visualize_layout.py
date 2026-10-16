#!/usr/bin/env python3
"""Draw a schematic top-down *layout*, not a MuJoCo rendering."""
import json
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle

ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT/'dimensions.json').read_text())
D = cfg['dimensions_m']
OX, OY = D['crate_center_x'], D['crate_center_y']
SX, SY = D['crate_depth_x'], D['crate_width_y']
BX, BY = D['bin_depth_x'], D['bin_width_y']
S = D['bin_pitch_y']
COL = {'red':'#C8313C','black':'#292C32','white':'#F9F6EF','cream':'#E9DCBA'}


def render(preset):
    fig, ax = plt.subplots(figsize=(10,6.8))
    ax.add_patch(Rectangle((.25,-.46), .92, .92, facecolor='#eeeeec',edgecolor='#999',linewidth=1))
    ax.add_patch(Rectangle((OX-SX/2, -SY/2), SX,SY, facecolor='#c6a070',edgecolor='#99764b',linewidth=5))
    bins={'black':+S,'white':0.,'red':-S}
    for name,y in bins.items():
        fc=COL[name]
        ax.add_patch(Rectangle((OX-BX/2,y-BY/2),BX,BY,facecolor=fc,alpha=.45,edgecolor=fc,linewidth=4))
        ax.text(OX+.08, y, f'{name.upper()} BIN',fontsize=9,color='#333',va='center')
    objects={
        'red_cuboid': ('black',-.021), 'black_cube':('black',+.027),
        'white_cylinder': ('white',-.021), 'white_cube':('white',+.027),
        'white_cuboid':('red',-.021),'red_cube':('red',+.027)
    }
    for name, spec in cfg['objects'].items():
        color=COL[spec['color']]
        if preset=='sorting' and name in ('red_cube','black_cube'):
            x,y=D['sorting_pickup_x'],(-D['sorting_pickup_y'] if name=='red_cube' else D['sorting_pickup_y'])
        else:
            bin_name, offs=objects[name]
            x,y=OX+offs,bins[bin_name]
        sz=spec['size_m']
        if name=='white_cylinder':
            ax.add_patch(Circle((x,y),sz[0]/2,facecolor=color,edgecolor='#777',linewidth=1.2))
        else:
            w,h=sz[:2]
            ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,facecolor=color,edgecolor='#777',linewidth=1.2))
        if preset=='sorting' and name in ('red_cube','black_cube'):
            ax.annotate(name, (x,y), xytext=(x-.03,y+(.05 if y<0 else -.06)),
                        arrowprops=dict(arrowstyle='-',color='#444'),fontsize=9)
    ax.set_xlim((.62 if preset=='photo' else .40),(.84 if preset=='photo' else .91))
    ax.set_ylim((-.16 if preset=='photo' else -.23),(.16 if preset=='photo' else .23))
    ax.set_aspect('equal');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)')
    ax.set_title(f'EG02ROBOT – {preset.upper()} preset (top view, schematic)')
    ax.grid(alpha=.15)
    fig.tight_layout()
    out=ROOT/'generated'/f'{preset}_schematic.png'
    fig.savefig(out,dpi=170)
    plt.close(fig)
    print(out)

for mode in ('photo','sorting'): render(mode)
