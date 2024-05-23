import os
import warnings
warnings.filterwarnings("ignore")
import gradio as gr
import mmCollector 


def on_connect_click(port_hori, port_vert):
    global recorders, available_ports
    try: 
        if port_hori and port_hori != 'None': 
            port_hori = available_ports[port_hori]
            recorders['hori'] = mmCollector.mmWaveRecoder(port=port_hori, device='iwr1843')
        if port_vert and port_vert != 'None': 
            port_vert = available_ports[port_vert]
            recorders['vert'] = mmCollector.mmWaveRecoder(port=port_vert, device='iwr1843')
        return 'Connect successful'
    except: 
        return 'Connect failed'

def on_set_click(num_frames): 
    global recorders, sending_status
    for rec in recorders.values():
        if rec is not None: 
            rec.send_config(num_frames)
    sending_status = False

def on_start_click(): 
    global recorders, sending_status
    if not sending_status: 
        for rec in recorders.values():
            if rec is not None: 
                rec.send_start()
    sending_status = True

def on_stop_click(): 
    global recorder_hori, recorder_vert, sending_status
    if sending_status: 
       for rec in recorders.values():
            if rec is not None: 
                rec.send_stop()
    sending_status = False
    
def on_configure_fpga(): 
    global recorders
    out = ''
    for id in recorders.keys():
        rec = recorders[id]
        if rec is not None:
            out += '[Output for {}]\n'.format(id)
            out += mmCollector.configure_fpga(id)
    return out

def on_listen_fpga(id): 
    global recorders
    rec = recorders[id]
    if rec is not None:
        mmCollector.start_record(id)
    return "Record done, check collected data"

def on_check_data(): 
    global recorders
    out = ''
    for id in recorders.keys():
        rec = recorders[id]
        if rec is not None:
            out += '[Output for {}]\n'.format(id)
            info = mmCollector.read_data(id)
            out += "File size: {} K\nStart time: {}\nEnd time: {}\nDuration: {}s\n".format(info['file_size'], info['start_time'], info['end_time'], info['duration'])
    return out

if __name__ == '__main__': 
    # variables
    available_ports = mmCollector.search_available_ports()
    recorders = {'hori': None, 'vert': None}
    sending_status = False
    
    with gr.Blocks() as demo:
        # connect to the mmWave sensor        
        dropdown_options = ['None'] + list(available_ports.keys())
        dropdown_hori = gr.Dropdown(choices=dropdown_options, label='Serial Port', info='Select the serial port of horizontal radar') 
        dropdown_vert = gr.Dropdown(choices=dropdown_options, label='Serial Port', info='Select the serial port of vertical radar') 
        btn_to_connect = gr.Button("Connect")
        out_of_connect = gr.Textbox(label='Connect Console', placeholder="", interactive=False)
            
        btn_to_configure = gr.Button('Configure FPGA')
        out_of_fpga_configure = gr.Textbox(label='Horizontal Console', placeholder="", interactive=False)
        with gr.Row(): 
            btn_to_listen_hori = gr.Button('Begin FPGA listening - Horizontal')
            btn_to_listen_vert = gr.Button('Begin FPGA listening - Vertical')
        
        slider_to_set = gr.Slider(minimum=0, maximum=600, value=50, label='Set the num frames to record')
              
        with gr.Row(): 
            btn_to_set = gr.Button('Send config')
            btn_to_start = gr.Button('Start record')
            btn_to_stop = gr.Button('Stop record')
            
        with gr.Column(): 
            btn_to_check = gr.Button('Check collected data')
            out_of_check = gr.Textbox(label='Data Info', placeholder="", interactive=False)
            
        # callback bind
        btn_to_connect.click(fn=on_connect_click, inputs=[dropdown_hori, dropdown_vert], outputs=out_of_connect)
        btn_to_set.click(fn=on_set_click, inputs=slider_to_set)
        btn_to_start.click(fn=on_start_click, inputs=None)
        btn_to_stop.click(fn=on_stop_click, inputs=None)
        # FPGA configuration 
        btn_to_configure.click(fn=on_configure_fpga, inputs=None, outputs=out_of_fpga_configure)
        # FPGA listening
        btn_to_listen_hori.click(fn=lambda x: on_listen_fpga('hori'), inputs=None)
        btn_to_listen_vert.click(fn=lambda x: on_listen_fpga('vert'), inputs=None)
        btn_to_check.click(fn=on_check_data, inputs=None, outputs=out_of_check)
    
    demo.launch()
