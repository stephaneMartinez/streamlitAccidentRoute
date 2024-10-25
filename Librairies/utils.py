from typing import Callable
import streamlit as st
import streamlit_antd_components as sac
import inspect, base64
from PIL import Image

image_404 = "donnees/images/404_page_inexistante.jpg"

FORMAT = [None, 'title', 'upper', "lambda x:f'A_{x}'"]
LABEL = [
    None,
    'label', 
    '**label**',
    '**label** <span class="badge rounded-pill badge-info">Info</span>'
]
MartineColor = ['dark', 'gray', 'red', 'pink', 'grape', 'violet', 'indigo', 'blue', 'cyan', 'teal', 'green', 'lime',
                'yellow', 'orange']
MartineSize = ['xs', 'sm', 'md', 'lg', 'xl']
AntColor = {
    'magenta': '#c41d7f',
    'red': '#cf1322',
    'volcano': '#d4380d',
    'orange': '#d46b08',
    'gold': '#d48806',
    'lime': '#7cb305',
    'green': '#389e0d',
    'cyan': '#08979c',
    'blue': '#0958d9',
    'geekblue': '#1d39c4',
    'purple': '#531dab'
}

size_help = 'support mantine size and int in px'
md_help = 'support str and markdown str'


def show_label(container=None, value='label', key=None):
    if container is not None:
        with container:
            return st.text_input('label', value, key=f'{key}-label', help=md_help)
    return st.text_input('label', value, key=f'{key}-label', help=md_help)


def show_description(container=None, value='', key=None):
    if container is not None:
        with container:
            return st.text_input('description', value, key=f'{key}-desc', help=md_help)
    return st.text_input('description', value, key=f'{key}-desc', help=md_help)


def show_index(container=None, option=None, index=0, key=None):
    if container is not None:
        with container:
            return st.selectbox('index', option, index=index, key=f'{key}-index')
    return st.selectbox('index', option, index=index, key=f'{key}-index')


def show_format_func(container=None, key=None):
    if container is not None:
        with container:
            return st.selectbox('format_func', FORMAT, 0, key=f'{key}-format', help='item label format')
    return st.selectbox('format_func', FORMAT, 0, key=f'{key}-format', help='item label format')


def show_color(index=0, label='color', options=(None, *MartineColor, '#4682b4', 'rgb(20,80,90)'),
               none_color='--primary-color', key=None, martine=True):
    colors = {'info': 'rgb(0, 66, 128)', 'success': 'rgb(23, 114, 51)', 'warning': 'rgb(146, 108, 5)',
              'error': 'rgb(125, 53, 59)', 'transparent': 'lightgray', None: none_color}
    btn = sac.buttons(
        items=[sac.ButtonsItem(
            label='None' if i is None else i,
            color=colors.get(i) if i in colors.keys() else (i if martine else AntColor.get(i, i))) for i in options
        ],
        label=label, index=index,
        size='xs', gap='xs', variant='filled', radius='lg', key=f'{key}-color'
    )
    return None if btn == 'None' else btn


def show_size(index=2, include_int=True, key=None):
    return st.radio('size', options=MartineSize + [25] if include_int else MartineSize, index=index, horizontal=True,
                    key=f'{key}-size', help=size_help)


def show_radius(index=2, key=None):
    return st.radio('radius', MartineSize + [20, 2], index=index, horizontal=True, key=f'{key}-radius', help=size_help)


def show_variant(options, index=0, key=None):
    return st.radio('variant', options, index=index, horizontal=True, key=f'{key}-variant')


def show_align(index=1, key=None):
    return st.radio('align', ['start', 'center', 'end'], index=index, horizontal=True, key=f'{key}-align')


def show_direction(index=0, key=None):
    return st.radio('direction', ["horizontal", "vertical"], index=index, horizontal=True, key=f'{key}-direction')


def show_checkbox(label, container=None, value=False, key=None):
    if container is not None:
        with container:
            return st.checkbox(label, value=value, key=f'{key}-{label}')
    return st.checkbox(label, value=value, key=f'{key}-{label}')


def show_radio(label=None, options=None, container=None, index=0, key=None):
    if container is not None:
        with container:
            return st.radio(label, options=options, index=index, horizontal=True, key=f'{key}-{label}')
    return st.radio(label, options=options, index=index, horizontal=True, key=f'{key}-{label}')


def show_space(number=3):
    st.markdown('<br>' * number, unsafe_allow_html=True)


def update_kw(d: dict, remove_keys: list = None):
    ff = d.get("format_func")
    if isinstance(ff, str) and 'lambda' in ff:
        d.update({"format_func": eval(ff)})
    label = d.get('label')
    if isinstance(label, str) and not label:
        d.update({'label': None})
    desc = d.get('description')
    if isinstance(desc, str) and not desc:
        d.update({'description': None})
    if remove_keys is not None:
        for i in remove_keys:
            if i in d.keys():
                del d[i]
    return d


def code_kw(kw: dict, _func):
    # func default kw
    default_kw = {
        k: v.default for k, v in inspect.signature(_func).parameters.items() if
        v.default is not inspect.Parameter.empty
    }
    # filter kw
    new_kw = {k: v for k, v in kw.items() if v != default_kw.get(k)}
    # dict to kw string
    return ", ".join(f'{k}={FORMAT[-1]}' if isinstance(v, Callable) else f'{k}={v!r}' for k, v in new_kw.items())


def show_code(x: str, open: bool = False):
    with st.expander('code', open):
        st.code(f'''
        import streamlit_antd_components as sac\n{x}
        ''', line_numbers=True)

def show_image(path, description=None, use_column_width=True):
    try :
        image = Image.open(path)

        # Afficher l'image
        st.image(image, caption=description, use_column_width=use_column_width)
    except :
        pass

def show_error_404 () :
    show_image(image_404, "Hum, c'est embarrassant")

# Fonction pour convertir une image locale en base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded