'''Python Class for constructing jupyter notebook widgets'''

import ipywidgets as widgets
import ipywidgets


class Widgets:
    def __init__(self):
        pass

    def dropdown(self, options: list, value: str, description: str):
        return ipywidgets.Dropdown(options=options,
                                   value=value,
                                   description=description,
                                   disabled=False)

    def int_slider(self, minimum: int, maximum: int, step: int, value: int, description: str):
        return widgets.IntSlider(
            value=value,
            min=minimum,
            max=maximum,
            step=step,
            description=description,
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )

    def checkbox(self, description: str):
        return widgets.Checkbox(
            value=False,
            description=description,
            disabled=False,
            indent=False
        )
