import os



class DataSaver:
    """数据保存器，负责将数据保存到文件"""
    def __init__(self, status=True):
        self.status = status
        if status:
            print("数据保存器启用")
        else:
            print("数据保存器禁用")
    
    def save_to_csv(self, df, path, filename):
        if not self.status:
            print("数据保存器已禁用，不保存数据")
            return False
        """保存DataFrame到CSV文件"""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        file_path = os.path.join(path, filename)
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"数据已保存到: {file_path}")
        return True