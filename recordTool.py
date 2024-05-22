import os
import gradio as gr
from mmCollector import mmWaveRecoder, search_available_ports


def on_connect_click(port_hori, port_vert):
    global recorder_hori, recorder_vert, available_ports
    try: 
        if port_hori and port_hori != 'None': 
            port_hori = available_ports[port_hori]
            recorder_hori = mmWaveRecoder(port=port_hori, device='iwr1642')
        if port_vert and port_vert != 'None': 
            port_vert = available_ports[port_vert]
            recorder_vert = mmWaveRecoder(port=port_vert, device='iwr1843')
        return 'Connect successful'
    except: 
        return 'Connect failed'


def on_start_click(): 
    global recorder_hori, recorder_vert, sending_status
    if not sending_status: 
        if recorder_vert is not None: 
            recorder_vert.send_start()
        if recorder_hori is not None: 
            recorder_hori.send_start()
        sending_status = True


def on_stop_click(): 
    global recorder_hori, recorder_vert, sending_status
    if sending_status: 
        if recorder_vert is not None: 
            recorder_vert.send_stop()
        if recorder_hori is not None: 
            recorder_hori.send_stop()
        sending_status = False
            
            
if __name__ == '__main__': 
    
    # variables
    recorder_hori = None
    recorder_vert = None
    selected_port_vert: mmWaveRecoder = None
    selected_port_hori: mmWaveRecoder = None
    available_ports = search_available_ports()
    sending_status = False
    
    with gr.Blocks() as demo:
        # connect to the mmWave sensor        
        with gr.Column(): 
            dropdown_options = ['None'] + list(available_ports.keys())
            dropdown_hori = gr.Dropdown(choices=dropdown_options, label='Serial Port', info='Select the serial port of horizontal radar') 
            dropdown_vert = gr.Dropdown(choices=dropdown_options, label='Serial Port', info='Select the serial port of vertical radar') 
            btn_to_connect = gr.Button("Connect")
            out_of_connect = gr.Textbox(label='Connect Console', placeholder="", interactive=False)
            
        with gr.Row(): 
            btn_to_start = gr.Button('Start record')
            btn_to_stop = gr.Button('Stop record')
            
        # callback bind
        btn_to_connect.click(fn=on_connect_click, inputs=[dropdown_hori, dropdown_vert], outputs=out_of_connect)
        btn_to_start.click(fn=on_start_click, inputs=None)
        btn_to_stop.click(fn=on_stop_click, inputs=None)
        
    demo.launch()
