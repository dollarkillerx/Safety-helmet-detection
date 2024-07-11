import socket
import mss
import numpy as np
import cv2
import pygetwindow as gw

def get_window_bbox(window_title):
    # 获取窗口的位置和尺寸
    window = gw.getWindowsWithTitle(window_title)[0]
    bbox = {'left': window.left, 'top': window.top, 'width': window.width, 'height': window.height}
    return bbox

def capture_and_send(window_title, server_ip, server_port):
    sct = mss.mss()
    bbox = get_window_bbox(window_title)

    # 创建socket连接
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))
    connection = client_socket.makefile('wb')

    try:
        while True:
            # 捕获屏幕
            sct_img = sct.grab(bbox)
            img = np.array(sct_img)

            # 转换为JPEG格式
            _, frame = cv2.imencode('.jpg', img)
            data = frame.tobytes()

            # 发送数据大小
            client_socket.sendall(len(data).to_bytes(4, byteorder='big'))
            # 发送图像数据
            client_socket.sendall(data)

            # 显示捕获的画面（可选）
            cv2.imshow('Captured Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        connection.close()
        client_socket.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    window_title = "Webex"  # 替换为目标应用的窗口标题
    server_ip = "127.0.0.1"  # 目标服务器IP地址
    server_port = 9999  # 目标服务器端口

    capture_and_send(window_title, server_ip, server_port)
