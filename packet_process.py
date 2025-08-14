import ipaddress
import json
import os
import pyshark
import tqdm

def layer_to_dic(layer):
    """
    Convert PFCP layer to a dictionary recursively.
    """
    
    if not hasattr(layer, 'field_names'):
        return str(layer)
    
    layer_dict = {}
    for field in layer.field_names:

        if field == "":
            continue
        
        field_value = getattr(layer, field)
        if isinstance(field_value, list):
            layer_dict[field] = [layer_to_dic(item) for item in field_value]
        else:
            if hasattr(field_value, 'field_names'):
                
                layer_dict[field] = layer_to_dic(field_value)
            else:
                # print(field,field_value)
                layer_dict[field] = str(field_value)
    return layer_dict
    
# read a pcapng file and print all info about the first 2 packets
def read_pcapng(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    try:
        capture = pyshark.FileCapture(file_path, display_filter='ip')
        packet_count = 0

        for packet in capture:
            if packet_count < 2:
                print(f"Packet {packet_count + 1}:")
                print(f"Timestamp: {packet.sniff_time}")
                print(f"Source IP: {packet.ip.src}")
                print(f"Destination IP: {packet.ip.dst}")
                print(f"Protocol: {packet.transport_layer}")
                print(f"Length: {packet.length} bytes")
                # If available, print additional fields
                if hasattr(packet, 'ip'):
                    print(f"Source Port: {packet[packet.transport_layer].srcport if hasattr(packet[packet.transport_layer], 'srcport') else 'N/A'}")
                    print(f"Destination Port: {packet[packet.transport_layer].dstport if hasattr(packet[packet.transport_layer], 'dstport') else 'N/A'}")
                print("-" * 40)
                # print payload if available
                if hasattr(packet, 'payload'):
                    print(f"Payload: {packet.payload}")
                print("=" * 40)
                packet_count += 1
            else:
                break

    except Exception as e:
        print(f"An error occurred while reading the pcapng file: {e}")
        
# read packets with payload and save to JSON
# the json format should be:
# {
#     "packets": [
#         {
#             "num" : "packet_num"
#             "src" : "source_ip",
#             "dst" : "destination_ip",
#             "protocol": "protocol_name",
#             "length": packet_length,
#             "src_port": source_port,
#             "dst_port": destination_port,
#             "payload": "packet_payload"
#         },
#         ...
#     ]
# }
def read_pcapng_with_payload(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    packets_data = {"packets": []}

    try:
        capture = pyshark.FileCapture(file_path, display_filter='ip')

        # for packet in capture:
        for packet in tqdm.tqdm(capture, desc="Processing packets", unit="packet"):
            # show current packet number
            # print(f"Processing packet: {packet.number}")
            # print(f"highest layer: {packet.highest_layer}")
            packet_payload = None
            pfcp_flag = False
            sctp_flag = False
            ngap_flag = False
            if hasattr(packet, 'tcp') and hasattr(packet.tcp, 'payload'):
                if packet.transport_layer != None:
                    transprort_layer = packet.transport_layer
                else:
                    transprort_layer = 'tcp'
                packet_payload = packet.tcp.payload
            # Check for UDP payload
            elif hasattr(packet, 'udp') and hasattr(packet.udp, 'payload'):
                if packet.transport_layer != None:
                    transprort_layer = packet.transport_layer
                else:
                    transprort_layer = 'udp'
                # print(f"Processing UDP packet: {packet.number}")
                packet_payload = packet.udp.payload
            elif hasattr(packet, 'pfcp') :
                if packet.transport_layer != None:
                    transprort_layer = packet.transport_layer
                else:
                    transprort_layer = 'pfcp'
                # print(f"Processing PFCP packet: {packet.number}")
                packet_payload = packet.pfcp
                pfcp_flag = True
            elif hasattr(packet, 'ngap'):
                if packet.transport_layer != None:
                    transprort_layer = packet.transport_layer
                else:
                    transprort_layer = 'sctp'
                print(f"Processing NGAP packet: {packet.number}")
                packet_payload = packet.ngap
                ngap_flag = True
                # print(packet_payload)
                
            elif hasattr(packet, 'sctp'):
                if packet.transport_layer != None:
                    transprort_layer = packet.transport_layer
                else:
                    transprort_layer = 'sctp'
                # print(f"Processing SCTP packet: {packet.number}")
                packet_payload = packet.sctp
                sctp_flag = True
            

            else:
                # print this packet is skipped due to unsupported protocol or no payload
                print(f"Skipping packet {packet.number} due to unsupported protocol or no payload")
                continue  # Skip packets without TCP/UDP payloads
            
            if packet_payload :
                # convert hex payload to ASCII, only if it is printable
                if not pfcp_flag and not sctp_flag and not ngap_flag:
                    ascii_payload = ''.join(chr(int(x, 16)) if 32 <= int(x, 16) <= 126 else '.' for x in packet_payload.split(':'))
                    # some of the packets may contain json data, so we need to check if it is valid JSON
                    # if it is valid JSON, we will parse it
                    try:
                        # stripped_payload = ascii_payload.strip().strip('.')
                        json_start = -1
                        json_end = -1
                        start_bracket = ascii_payload.find('[')
                        start_brace = ascii_payload.find('{')
                        if start_bracket != -1 and (start_brace == -1 or start_brace > start_bracket):
                            json_start = start_bracket
                            json_end = ascii_payload.find(']', json_start)
                        elif start_brace != -1:
                            json_start = start_brace
                            # find the last { in the payload
                            json_end = ascii_payload.rfind('}', json_start)


                        if json_start != -1:
                            json_str = ascii_payload[json_start:json_end + 1]
                            ascii_payload = json.loads(json_str)
                            # print this packet is processed
                            print(f"Processing packet {packet.number} with JSON payload")

                        # if there is no valid JSON, skip this packet
                        if json_start == -1 or json_end == -1:
                            # print this packet is skipped due to no valid JSON
                            print(f"Skipping packet {packet.number} due to no valid JSON")
                            continue
                    except json.JSONDecodeError:
                        # print this packet is skipped due to no valid JSON
                        print(f"Skipping packet {packet.number} due to no valid JSON")
                        # pass  # Not valid JSON, keep as is
                        continue  # Skip this packet if it is not valid JSON
                else:
                    if hasattr(packet, 'pfcp'):
                        # print this packet is PFCP packet
                        print(f"Processing PFCP packet: {packet.number}")
                        ascii_payload = layer_to_dic(packet.pfcp)

                    elif hasattr(packet, 'ngap'):
                        # print this packet is NGAP packet
                        print(f"Processing NGAP packet: {packet.number}")
                        ascii_payload = layer_to_dic(packet.ngap)
                        # print(ascii_payload)

                    elif hasattr(packet, 'sctp'):
                        # print this packet is SCTP packet
                        print(f"Processing SCTP packet: {packet.number}")
                        ascii_payload = layer_to_dic(packet.sctp)

                    else:
                        ascii_payload = {}
                packet_info = {
                    "num": packet.number,
                    "src": packet.ip.src,
                    "dst": packet.ip.dst,
                    "protocol": packet.highest_layer,
                    "length": int(packet.length),
                    "src_port": packet[transprort_layer].srcport if hasattr(packet[transprort_layer], 'srcport') else None,
                    "dst_port": packet[transprort_layer].dstport if hasattr(packet[transprort_layer], 'dstport') else None,
                    "payload": ascii_payload
                }
                packets_data["packets"].append(packet_info)
            


    except Exception as e:
        print(f"An error occurred while reading the pcapng file: {e}")

    # how many packets are processed
    print(f"Total packets processed: {len(packets_data['packets'])}")
    # Save to JSON file
    with open('packets_data.json', 'w') as json_file:
        json.dump(packets_data, json_file, indent=4)
        print("Packet data saved to packets_data.json")
        
    capture.close()  # Close the capture to free resources

        
# Example usage
if __name__ == "__main__":
    pcapng_file = 'pcap/5GC.pcapng'  # Replace with your pcapng file path
    # read_pcapng(pcapng_file)
    read_pcapng_with_payload(pcapng_file)
    
    # capture = pyshark.FileCapture(pcapng_file)
    # packet = capture[106]
    # # Force dissection
    # _ = packet.layers 

    # # --- New Debugging Steps ---
    # print("-" * 20)
    # print("Debugging Packet 107:")
    # print(f"Packet object representation:\n{packet}")
    # print(packet.ngap)
    # print(f"List of all layers found: {packet.layers}")
    # print(layer_to_dic(packet.ngap))
    # print(f"Value of transport_layer attribute: {packet.transport_layer}")
    # print("-" * 20)
    # # --- End Debugging Steps ---

    # print(packet['sctp'].srcport)  # Accessing SCTP source port
    # capture.close()
