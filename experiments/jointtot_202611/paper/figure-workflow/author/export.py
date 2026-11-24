"""Render the current Pixelmator SVG into a text/vector, full-width paper PDF.
Uses system librsvg and Cairo (no Python packages or network). Never writes the SVG.
"""
from pathlib import Path
import ctypes as C
from ctypes.util import find_library
import os
import tempfile
import xml.etree.ElementTree as ET

D=Path(__file__).resolve().parent

def export():
    src=D/'jointtot-workflow-en.svg'; dst=src.with_suffix('.pdf'); raw=src.read_bytes()
    root=ET.fromstring(raw); x,y,w,h=map(float,root.attrib['viewBox'].split());assert x==y==0 and w>0 and h>0
    size=178*72/25.4; height=size*h/w
    if Path('/etc/fonts/fonts.conf').is_file():os.environ.setdefault('FONTCONFIG_FILE','/etc/fonts/fonts.conf')
    cairo=C.CDLL(find_library('cairo'));rsvg=C.CDLL(find_library('rsvg-2'));go=C.CDLL(find_library('gobject-2.0'))
    P,S,I,F=C.c_void_p,C.c_char_p,C.c_int,C.c_double
    class Rect(C.Structure): _fields_=[(x,F) for x in ('x','y','width','height')]
    def fn(lib,name,ret,*args):
        f=getattr(lib,name);f.argtypes=args;f.restype=ret;return f
    load=fn(rsvg,'rsvg_handle_new_from_data',P,S,C.c_size_t,P);render=fn(rsvg,'rsvg_handle_render_document',I,P,P,C.POINTER(Rect),P)
    surface=fn(cairo,'cairo_pdf_surface_create',P,S,F,F);meta=fn(cairo,'cairo_pdf_surface_set_metadata',None,P,I,S)
    context=fn(cairo,'cairo_create',P,P);finish=fn(cairo,'cairo_surface_finish',None,P);stat=fn(cairo,'cairo_surface_status',I,P)
    free_context=fn(cairo,'cairo_destroy',None,P);free_surface=fn(cairo,'cairo_surface_destroy',None,P);unref=fn(go,'g_object_unref',None,P)
    handle=load(raw,len(raw),None);assert handle,'Invalid SVG'
    with tempfile.NamedTemporaryFile(dir=D,suffix='.pdf') as tmp:
        surf=surface(os.fsencode(tmp.name),size,height);ctx=context(surf)
        try:
            meta(surf,0,b'JointToT: Search workflow');meta(surf,5,b'')
            assert render(handle,ctx,C.byref(Rect(0,0,size,height)),None),'SVG render failed'
            finish(surf);assert not stat(surf),'PDF export failed'
        finally:
            free_context(ctx);free_surface(surf);unref(handle)
        data=Path(tmp.name).read_bytes();assert data.startswith(b'%PDF-') and data.rstrip().endswith(b'%%EOF')
        if not dst.exists() or dst.read_bytes()!=data: dst.write_bytes(data)
    print(f'{dst.name} ({size*25.4/72:.0f} × {height*25.4/72:.1f} mm)')

if __name__=='__main__':export()
