import asyncio
import websockets
import json
import requests
import tempfile
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

async def test_websocket_logs():
    """测试WebSocket日志功能"""
    print("=== 测试WebSocket日志功能 ===")
    
    try:
        # 连接WebSocket
        async with websockets.connect(WS_URL) as websocket:
            print("✅ WebSocket连接成功")
            
            # 创建测试文件
            test_content = """
            这是一个测试文档，用于验证WebSocket日志功能。
            
            # HyperRAG 日志测试
            
            本文档将被上传并嵌入到HyperRAG系统中，期间会产生详细的处理日志。
            
            ## 测试内容
            
            1. 文档上传测试
            2. 文件解析测试  
            3. 文档嵌入测试
            4. WebSocket日志传输测试
            
            这是完整的测试流程，用于验证实时日志显示功能。
            """
            
            # 保存为临时文件
            test_file_path = "websocket_test.txt"
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            try:
                # 1. 上传文件
                print("\n1. 上传测试文件...")
                with open(test_file_path, 'rb') as f:
                    files = {'files': (test_file_path, f, 'text/plain')}
                    response = requests.post(f"{BASE_URL}/files/upload", files=files)
                
                if response.status_code != 200:
                    print(f"❌ 文件上传失败: {response.text}")
                    return
                
                files_data = response.json().get('files', [])
                if not files_data:
                    print("❌ 没有获取到上传的文件信息")
                    return
                
                file_id = files_data[0].get('file_id')
                print(f"✅ 文件上传成功，ID: {file_id}")
                
                # 2. 启动嵌入并监听日志
                print("\n2. 启动文档嵌入并监听日志...")
                embed_data = {
                    "file_ids": [file_id],
                    "chunk_size": 500,
                    "chunk_overlap": 100
                }
                
                response = requests.post(f"{BASE_URL}/files/embed-with-progress", json=embed_data)
                if response.status_code != 200:
                    print(f"❌ 嵌入启动失败: {response.text}")
                    return
                
                print("✅ 嵌入处理已启动，开始监听日志...")
                
                # 3. 监听WebSocket消息
                log_count = 0
                progress_count = 0
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        msg_type = data.get('type', 'unknown')
                        
                        if msg_type == 'log':
                            log_count += 1
                            timestamp = data.get('timestamp', 0)
                            level = data.get('level', 'INFO')
                            log_message = data.get('message', '')
                            
                            print(f"📝 [{level}] {log_message}")
                            
                        elif msg_type == 'progress':
                            progress_count += 1
                            current = data.get('current', 0)
                            total = data.get('total', 0)
                            percentage = data.get('percentage', 0)
                            message = data.get('message', '')
                            
                            print(f"📊 进度: {current}/{total} ({percentage:.1f}%) - {message}")
                            
                        elif msg_type == 'file_processing':
                            filename = data.get('filename', '')
                            stage = data.get('stage', '')
                            message = data.get('message', '')
                            print(f"🔄 处理: {filename} - {stage} - {message}")
                            
                        elif msg_type == 'file_completed':
                            filename = data.get('filename', '')
                            print(f"✅ 完成: {filename}")
                            
                        elif msg_type == 'all_completed':
                            print(f"🎉 所有文档处理完成!")
                            break
                            
                        elif msg_type == 'error' or msg_type == 'file_error':
                            error = data.get('error', 'Unknown error')
                            print(f"❌ 错误: {error}")
                            break
                            
                    except json.JSONDecodeError:
                        print(f"⚠️  收到非JSON消息: {message}")
                
                print(f"\n📊 统计信息:")
                print(f"   - 收到日志消息: {log_count} 条")
                print(f"   - 收到进度消息: {progress_count} 条")
                
                # 4. 清理测试文件
                print("\n4. 清理测试数据...")
                delete_response = requests.delete(f"{BASE_URL}/files/{file_id}")
                if delete_response.status_code == 200:
                    print("✅ 测试文件删除成功")
                else:
                    print(f"⚠️  测试文件删除失败: {delete_response.text}")
                
            finally:
                # 清理本地临时文件
                if os.path.exists(test_file_path):
                    os.remove(test_file_path)
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

def main():
    """主函数"""
    print("开始测试WebSocket日志功能...")
    print("请确保后端服务已启动 (uvicorn main:app --reload)")
    
    try:
        asyncio.run(test_websocket_logs())
        print("\n🎉 WebSocket日志功能测试完成!")
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")

if __name__ == "__main__":
    main() 