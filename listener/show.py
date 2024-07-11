import socket
import cv2
import numpy as np

def receive_and_display(server_ip, server_port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, server_port))
    server_socket.listen(1)
    conn, _ = server_socket.accept()

    try:
        while True:
            # 接收数据大小
            data_len = int.from_bytes(conn.recv(4), byteorder='big')
            data = b''

            # 接收图像数据
            while len(data) < data_len:
                packet = conn.recv(4096)
                if not packet:
                    break
                data += packet

            # 将数据转换为图像
            frame = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            # 显示图像
            cv2.imshow('Received Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        conn.close()
        server_socket.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    server_ip = "0.0.0.0"  # 服务器IP地址
    server_port = 9999  # 服务器端口

    receive_and_display(server_ip, server_port)
