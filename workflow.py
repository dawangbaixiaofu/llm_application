from concurrent.futures import ThreadPoolExecutor
import json
import time


# config
general_prompt_template = "write prompt template here for general module."
industry_period_template = "write prompt template here for industry period module."
enterprise_wechat_template = "write prompt template here for enterprise wechat module."

module_params = [
    {
        'module_name': 'general',
        'prompt_template': general_prompt_template,
        'nums': 5,
        'api_params': None
    },

    {
        'module_name': 'industry_period',
        'prompt_template': industry_period_template,
        'nums': 4,
        'api_params': None
    },

    {
        'module_name': 'enterprise_wechat',
        'prompt_template': enterprise_wechat_template,
        'nums': 3,
        'api_params': None
    }
]


def make_prompt(enterprise_name:str, base_rate:float, prompt:str) -> str:
    prompt = prompt.format(enterprise_name, base_rate)
    return prompt


def agent_generate_simulate(prompt:str) -> str:
    """
    模拟调用LLM生成结果
    :param prompt: 提示语
    :return: 生成的结果
    """

    time.sleep(5)  # 模拟模型延迟

    return prompt


def call_agent_parallel(module_params:list) -> dict:
    """
    对每一个模块，使用多线程调用agent生成结果
    :param module_params:l
        module_name: 模块名称
        prompt_template: 提示词模板
        prompt: 提示词
        nums: 调用agent次数
    :return: dict 返回值：{module_name: [generation1, generation2, ...], ..., }
    """
    total_nums = 0
    for module in module_params:
        total_nums += module['nums']
    
    res = {}
    module_futures = {}
    with ThreadPoolExecutor(max_workers=total_nums) as executor:
        for module in module_params:
            if 'prompt' not in module:
                raise Exception("must make prompt for this module before you call agent api.")
            module_name = module['module_name']
            prompt = module['prompt']
            nums = module['nums']
            futures = [executor.submit(agent_generate_simulate, prompt) for _ in range(nums)]
            module_futures[module_name] = futures
        
        for module_name, futures in module_futures.items():
            res[module_name] = [future.result() for future in futures]
    return res
    


def merge_factor(generations:list) -> dict:
    """
    合并影响因子
    :param generations: 生成的结果列表
    :return: 合并后的影响因子，数据结构：{"factor_name": [impact1, impact2, ...], ...}
    """
    # 合并facotr
    merged_factors = {}

    for generation in generations:
        try:
            data = json.loads(generation)
            if "adjustments" in data:
                for adjustment in data["adjustments"]:
                    factor_name = adjustment["factor"]
                    impact = float(adjustment["impact"])
                    if factor_name not in merged_factors:
                        merged_factors[factor_name] = [impact]
                    else:
                        merged_factors[factor_name].append(impact)
        except json.JSONDecodeError:
            print("JSON解码错误")
    
    # fill value 0 for ungiven factor genereated by llm
    _len = len(generations)
    for factor in merged_factors:
        if len(merged_factors[factor]) < _len:
            merged_factors[factor] += [0] * (_len - len(merged_factors[factor]))

    return merged_factors



def extract_factor(generation:str) -> dict:
    """
    提取影响因子
    :param generation: 生成的结果
    :return: 影响因子{"factor_name": impact, ...}
    """
    try:
        data = json.loads(generation)
        if "adjustments" in data:
            factors = {}
            for adjustment in data["adjustments"]:
                factor_name = adjustment["factor"]
                impact = float(adjustment["impact"])
                factors[factor_name] = float(impact)
            return factors
        else:
            raise ValueError("生成结果中不包含调整因子")
    except json.JSONDecodeError:
        print("JSON解码错误")
        return {}



def filter_factor(generations:list, threshold: float) -> dict:
    """
    过滤出影响因子
    :param generations: 生成的结果列表
    :return: 影响因子列表
    """
    
    # 合并facotr
    merged_factors = merge_factor(generations)
    
    # filter out factor according to coeffieient of variance
    filtered_factors = {}
    for factor_name, impacts in merged_factors.items():
        mean_impact = sum(impacts) / len(impacts)
        variance = sum((x - mean_impact) ** 2 for x in impacts) / len(impacts)
        std_dev = variance ** 0.5
        coeff_of_variance = std_dev / mean_impact if mean_impact != 0 else 10
        if coeff_of_variance <= threshold:
            filtered_factors[factor_name] = mean_impact
    return filtered_factors





def cal_final_rate(base_rate:float, factors:dict) -> float:
    """
    计算最终利率
    :param base_rate: 基础利率
    :param filtered_factors: 过滤后的影响因子列表
    :return: 最终利率
    """
    final_rate = base_rate
    for factor, impact in factors.items():
        if isinstance(impact, float):
            final_rate += impact
        elif isinstance(impact, list):
            final_rate += sum(impact) / len(impact)
        else:
            raise ValueError("影响因子必须是float或list类型")
    return final_rate


def filter_factor_for_ent_wechat() -> dict:
    # todo: filter rule for enterprise wechat conversation
    pass




def workflow(enterprise_name:str, base_rate:float, module_params:list, threshold:float) -> dict:
    """
    主工作流函数
    :param enterprise_name: 企业名称
    :param base_rate: 基础利率
    :param module_params：各个定价模块的参数
    :param threshold: 过滤阈值
    :return dict: 最终利率和影响因子等
    """
    # make prompt for each module
    for module in module_params:
        prompt_template = module['prompt_template']
        module['prompt'] = make_prompt(enterprise_name, base_rate, prompt_template)

        
    # call agent parallel for all modules
    module_generations:dict = call_agent_parallel(module_params)

    # filter out factors for 'general' and 'industry_period' module using filter rule 1.
    module_filtered_factors = {}
    for module, generations in module_generations.items():
        if module in ['general', 'industry_period']:
            module_filtered_factors[module] = filter_factor(generations, threshold)
        elif module == 'enterprise_wechat':
            module_filtered_factors[module] = filter_factor_for_ent_wechat()
        else:
            raise Exception(f"'{module}' module name is wrong.")
    
    # concatenate all flitered factors
    total_factors = {}
    for module, filtered_factors in module_filtered_factors.items():
        total_factors.update(filtered_factors)

    # calculate final risk rate
    final_rate = cal_final_rate(base_rate, total_factors)
    
    return {
        "enterprise_name": enterprise_name,
        "base_rate": base_rate,
        "final_rate": final_rate,
        "factors": total_factors,
    }

if __name__ == "__main__":
    # to test consumed time of calling agent parallel
    start = time.time()

    # make prompt for each module
    enterprise_name = 'xxx'
    base_rate = 5.6
    for module in module_params:
        prompt_template = module['prompt_template']
        module['prompt'] = make_prompt(enterprise_name, base_rate, prompt_template)

    module_generations:dict = call_agent_parallel(module_params)
    
    end = time.time()
    duration = end - start
    print(f"total time consumed: {duration} s")
    print(module_generations)


    
