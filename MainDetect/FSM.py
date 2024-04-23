class AssemblyProcessStateMachine:
    def __init__(self):
        # 定义装配过程的各个阶段
        self.states = {
            0: "初始化",
            1: "检测零件1",
            2: "检测零件2",
            3: "检测零件3",
            4: "检测零件4",
            5: "完成装配"
        }
        self.current_state = 0  # 初始状态为初始化

    def update_state(self, det, acf, ang=None):
        # 检测装配位置错误的条件
        if self.current_state > 1 and not self.check_assembly_position(det, ang):
            return "装配位置错误"

        # 根据当前状态和输入数据更新状态
        if self.current_state == 0:  # 初始化状态
            if self.check_part_detection(det, acf[0]):
                self.current_state = 1
        elif self.current_state < 4:  # 检测各个零件的状态
            if self.check_part_detection(det, acf[self.current_state]):
                self.current_state += 1
        elif self.current_state == 4:  # 检测完成，进入完成装配状态
            if self.check_part_detection(det, acf[self.current_state]):
                self.current_state = 5

        return self.states[self.current_state]

    def check_part_detection(self, det, acf_threshold):
        # 在实际应用中，你需要根据具体的检测逻辑来判断零件是否被检测到
        # 这里只是一个示例，假设检测到的零件数量大于等于1即认为检测到了
        for part in det:
            if len(part) >= 1:
                if max(part[0]) >= acf_threshold:
                    return True
        return False

    def check_assembly_position(self, det, ang):
        # 在实际应用中，你需要根据具体的逻辑来判断装配位置是否正确
        # 这里只是一个示例，假设通过比较角度来判断装配位置是否正确
        # 如果角度符合要求，则认为装配位置正确，否则认为位置错误
        # 你可能需要根据实际情况修改此处的逻辑
        return ang == desired_angle


# 初始化状态机
assembly_process_sm = AssemblyProcessStateMachine()

# 示例数据
acf = [0.9, 0.8, 0.85, 0.7]  # 四个动作的置信度
det = [[(0.95, 0.1)], [(0.9, 0.2)], [(0.85, 0.3)], [], [(0.7, 0.5)]]  # 部件检测结果
ang = 90  # 关键部件的角度

# 更新状态并获取当前状态
current_state = assembly_process_sm.update_state(det, acf, ang)
print("当前装配阶段：", current_state)
