from utils.DB import DbHelper

# 根据Task ID 从 cc_Task 中查找 Task排序类型 1：手动已排序 2：需要自动排序
def select_task(taskID):
    sql = "select TaskType from cc_TaskDetail where ID='taskID'"
    result = DbHelper.select_sql(sql)
    return result


# 根据Task ID 从 cc_TaskDetail 中查找 Task信息
def select_task_detail(taskID):
    sql = "select * from cc_TaskDetail where TaskID='taskID'"
    result = DbHelper.select_sql(sql)
    return result

def select_nest_info_by_id(id):
    '''
    The output dict key is NestID
    :param id:
    :return:
    '''
    sql = "select * from cc_NestInfo where Id=\'{}\'".format(id)
    results = DbHelper.select_sql(sql)
    output = {}
    for result in results:
        local = {
            # 'Id': result[0],
            'NestID': result[1],
            'GroupID': result[2],
            # 'Rate': result[3],
            # 'FactoryID': result[4],
            # 'ExMaterialID': result[5],
            # 'MaterielName': result[6],
            # 'MaterielTypeID': result[7],
            # 'Texture': result[8],
            'Thickness': result[9],
            'Height': result[10],
            'Width': result[11],
            # 'MTWeight': result[12],
            # 'NestDate': result[13],
            # 'BookSheet': result[14],
            # 'OptionID': result[15],
            # 'RequireDoneDate': result[16],
            'LocalLink': result[25],
        }
        output[local['NestID']] = local
    return output

def select_nest_info_by_nestID(nestID):
    sql = "select * from cc_NestInfo where NestID=\'{}\'".format(nestID)
    results = DbHelper.select_sql(sql)
    output = {}
    for result in results:
        local = {
            # 'Id': result[0],
            'NestID': result[1],
            'GroupID': result[2],
            # 'Rate': result[3],
            # 'FactoryID': result[4],
            # 'ExMaterialID': result[5],
            # 'MaterielName': result[6],
            # 'MaterielTypeID': result[7],
            # 'Texture': result[8],
            'Thickness': result[9],
            'Height': result[10],
            'Width': result[11],
            # 'MTWeight': result[12],
            # 'NestDate': result[13],
            # 'BookSheet': result[14],
            # 'OptionID': result[15],
            # 'RequireDoneDate': result[16],
            'LocalLink': result[25],
        }
        output[local['NestID']] = local
    return output

def select_part_info_by_nestID(nestID) -> dict:
    '''
    The output dict key is PartID
    :param nestID:
    :return: dict
    '''
    sql = "select * from cc_PartInfo where NestID=\'{}\'".format(nestID)
    results = DbHelper.select_sql(sql)
    output = {}
    for result in results:
        local = {
            # 'Id': result[0],
            'NestID': result[1],
            'PartID': result[3],
            # 'PartWidth': result[6],  # 使用从dxf解析出的图形尺寸，不使用表格中的尺寸
            # 'PartHeight': result[7],
            'PartSN': result[8],
            # 'Technics': int(result[9]) if result[9] is not None and result[9] != '' else 1,
            # 'NestPlanID': int(result[10]) if result[10] is not None and result[10] != '' else 1,
            'Technics': result[9] if result[9] is not None and result[9] != '' else '1',
            'NestPlanID': result[10] if result[10] is not None and result[10] != '' else '1',
            'RequireSort': result[23],
        }
        output[local['PartID']] = local
    return output

def select_result_all_from_table(table):
    sql = "select * from {}".format(table)
    result = DbHelper.select_sql(sql)
    if result is None:
        result = []
    return result

def select_alignment_result_by_nestID(nestID):
    sql = "select * from cc_SortAlignmentResult where NestID='{}'".format(nestID)
    results = DbHelper.select_sql(sql)
    output = {}
    for result in results:
        local = {
            'NestID': result[1],
            'DeviceCode': result[2],
            'AlignmentResult': result[3],
            'Offset_X': result[4],
            'Offset_Y': result[5],
            'Offset_A': result[6],
        }
        output[local['NestID']] = local
    return output[nestID]


def insert_strategy_result(nestID, type, status, comment=""):
    sql = "insert into cc_StrategyResult (NestID, StrategyType, StrategyStatus, Comment)" \
          " VALUES(\'{}\', {}, {}, \'{}\')".format(nestID, type, status, comment)
    result = DbHelper.insert_sql(sql)
    return result


# 生成cc_SortStrategy记录
def insert_sort_strategy(strategy):
    sql = "insert into cc_SortStrategy (NestID, GrabTimes, GrabIndex, RobotIndex, Origin_X, Origin_Y, Origin_A, " \
          "Destination_X, Destination_Y, Destination_A, Magnetic50, Magnetic100, Power50, Power100, BatchIndex, BatchQuantity) " \
          "VALUES (\'{}\', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
        strategy["NestID"], strategy["GrabTimes"], strategy["GrabIndex"], strategy["RobotIndex"],
        strategy["Origin_X"], strategy["Origin_Y"], strategy["Origin_A"], strategy["Destination_X"],
        strategy["Destination_Y"], strategy["Destination_A"], strategy["Magnetic50"], strategy["Magnetic100"],
        strategy["Power50"], strategy["Power100"], strategy["BatchIndex"], strategy["BatchQuantity"],
    )
    result = DbHelper.insert_sql(sql)
    return result

# 生成cc_SortStrategyOffset记录
def insert_sort_strategy_offset(strategy):
    sql = "insert into cc_SortStrategyOffset (NestID, GrabTimes, GrabIndex, RobotIndex, Origin_X, Origin_Y, Origin_A, " \
          "Destination_X, Destination_Y, Destination_A, Magnetic50, Magnetic100, Power50, Power100, BatchIndex, BatchQuantity) " \
          "VALUES (\'{}\', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
        strategy["NestID"], strategy["GrabTimes"], strategy["GrabIndex"], strategy["RobotIndex"],
        strategy["Origin_X"], strategy["Origin_Y"], strategy["Origin_A"], strategy["Destination_X"],
        strategy["Destination_Y"], strategy["Destination_A"], strategy["Magnetic50"], strategy["Magnetic100"],
        strategy["Power50"], strategy["Power100"], strategy["BatchIndex"], strategy["BatchQuantity"],
    )
    result = DbHelper.insert_sql(sql)
    return result

# 生成cc_StackStrategy记录
def insert_stack_strategy(strategy, batchIndex = None):
    sql = "insert into cc_StackStrategy (NestID, GrabTimes, GrabIndex, RobotIndex, Origin_X, Origin_Y, Origin_A, " \
          "Destination_X, Destination_Y, Destination_A, Magnetic50, Magnetic100, Power50, Power100, StockIndex) " \
          "VALUES (\'{}\', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, \'{}\')".format(
        strategy["NestID"], strategy["GrabTimes"], strategy["GrabIndex"], strategy["RobotIndex"],
        strategy["Origin_X"], strategy["Origin_Y"], strategy["Origin_A"], strategy["Destination_X"],
        strategy["Destination_Y"], strategy["Destination_A"], strategy["Magnetic50"], strategy["Magnetic100"],
        strategy["Power50"], strategy["Power100"], strategy["StockIndex"],
    )
    result = DbHelper.insert_sql(sql)
    return result

# 生成cc_StackStrategyOffset记录

def delete_stack_strategy_offset(nestID, batchIndex):
    sql="delete from cc_StackStrategyOffset where NestID=\'{}\' and BatchIndex={}".format(
        nestID,
        batchIndex)
    result = DbHelper.delete_sql(sql)
    return result



def insert_stack_strategy_offset(strategy, batchIndex = None):
    sql = "insert into cc_StackStrategyOffset (NestID, GrabTimes, GrabIndex, RobotIndex, Origin_X, Origin_Y, Origin_A, " \
          "Destination_X, Destination_Y, Destination_A, Magnetic50, Magnetic100, Power50, Power100, StockIndex, BatchIndex) " \
          "VALUES (\'{}\', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, \'{}\', {})".format(
        strategy["NestID"], strategy["GrabTimes"], strategy["GrabIndex"], strategy["RobotIndex"],
        strategy["Origin_X"], strategy["Origin_Y"], strategy["Origin_A"], strategy["Destination_X"],
        strategy["Destination_Y"], strategy["Destination_A"], strategy["Magnetic50"], strategy["Magnetic100"],
        strategy["Power50"], strategy["Power100"], strategy["StockIndex"], batchIndex,
    )
    result = DbHelper.insert_sql(sql)
    return result


 # 根据排序结果 更新 cc_TaskDetail 中切割任务的执行顺序
def update_task_index_detail(recordID, nestIndex):
    sql = "update cc_TaskDetail set  NestIndex = nestIndex where ID='recordID'"
    DbHelper.update_sql(sql)


#  将下料策略存入数据库

# inser test sql
def insert_test(name, age):
    sql = "insert into cc_test (Name, Age) VALUES ('{0}', {1})".format(name, age)
    DbHelper.insert_sql(sql)


if __name__ == "__main__":
    # insert_test("Shenghao", 18)
    # nest_info = select_nest_info('3210519A0875')
    # part_info = select_part_info_by_nestID('3210519A0875')

    # nest_info = select_nest_info_by_id(1)
    # part_info = select_part_info_by_nestID(list(nest_info.keys())[0])
    # print(nest_info)
    # print(part_info)

    # results_old = select_result_all_from_table("cc_StrategyResult")
    # result = insert_strategy_result(len(results_old)+1, "blabla", 0, 1, "babulibaba")

    # result = insert_sort_strategy(
    #     {
    #         "NestID": "2100000",
    #         "GrabTimes": 10,
    #         "GrabIndex": 1,
    #         "RobotIndex": 32,
    #         "Origin_X": 10,
    #         "Origin_Y": 10,
    #         "Origin_A": 10,
    #         "Destination_X": 10,
    #         "Destination_Y": 10,
    #         "Destination_A": 10,
    #         "Magnetic50": 10,
    #         "Magnetic100": 10,
    #         "Power50": 1,
    #         "Power100": 1,
    #     }, len(select_result_all_from_table("cc_SortStrategy"))+1
    # )

    # result = insert_sort_strategy(
    #     {
    #         "NestID": "2100000",
    #         "GrabTimes": 10,
    #         "GrabIndex": 1,
    #         "RobotIndex": 32,
    #         "Origin_X": 10,
    #         "Origin_Y": 10,
    #         "Origin_A": 10,
    #         "Destination_X": 10,
    #         "Destination_Y": 10,
    #         "Destination_A": 10,
    #         "Magnetic50": 10,
    #         "Magnetic100": 10,
    #         "Power50": 1,
    #         "Power100": 1,
    #     }, len(select_result_all_from_table("cc_SortStrategy"))+1
    # )

    result = insert_stack_strategy(
        {
            "NestID": "2100000",
            "GrabTimes": 10,
            "GrabIndex": 1,
            "RobotIndex": 32,
            "Origin_X": 10,
            "Origin_Y": 10,
            "Origin_A": 10,
            "Destination_X": 10,
            "Destination_Y": 10,
            "Destination_A": 10,
            "Magnetic50": 10,
            "Magnetic100": 10,
            "Power50": 1,
            "Power100": 1,
            "StockIndex": 1
        }, len(select_result_all_from_table("cc_StackStrategy"))+1)