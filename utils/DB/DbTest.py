import DbHelper

# 给出一个TaskId,如果 TaskType=自动排序 Pyton 计算这个 TaskId 下所有 TaskDetail 的排序结果并更新数据库
def method1():
    return 1

# 查询 cc_CurrentTaskDetail 表下所有 NestId, 按照 NestId 所属的 GroupID （cc_NestInfo） 计算分拣策略和下料策略 并记录到cc_SortStrategy 和 cc_StackStrategy
def method2():
    return 1

# 查询 cc_CurrentTaskDetail 表下所有 NestId, 按照 NestId 所属的 GroupID （cc_NestInfo） 计算分拣策略和下料策略 并记录到cc_SortStrategy 和 cc_StackStrategy
def method3():
    return 1

# 根据 NestId 查询 cc_SortAlignmentResult 对应的记录, 并计算分拣策略 并记录到cc_SortStrategyOffset 中，并将计算结果插入 cc_StrategyResult
def method4():
    return 1

# 根据 NestId 读取指定位置的 png 图片（记录UpdateTime时间 请求时间10min 内最新的一条）, 并计算分拣策略 并记录到cc_SortStrategyOffset 中
def method5():
    return 1