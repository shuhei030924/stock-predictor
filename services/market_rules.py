
def get_price_limits(reference_price: float) -> tuple[float, float]:
    """
    日本株の値幅制限（ストップ高・ストップ安）を計算する
    
    Args:
        reference_price (float): 基準値段（通常は前日終値）
        
    Returns:
        tuple[float, float]: (ストップ安価格, ストップ高価格)
    """
    if reference_price < 100:
        limit = 30
    elif reference_price < 200:
        limit = 50
    elif reference_price < 500:
        limit = 80
    elif reference_price < 700:
        limit = 100
    elif reference_price < 1000:
        limit = 150
    elif reference_price < 1500:
        limit = 300
    elif reference_price < 2000:
        limit = 400
    elif reference_price < 3000:
        limit = 500
    elif reference_price < 5000:
        limit = 700
    elif reference_price < 7000:
        limit = 1000
    elif reference_price < 10000:
        limit = 1500
    elif reference_price < 15000:
        limit = 3000
    elif reference_price < 20000:
        limit = 4000
    elif reference_price < 30000:
        limit = 5000
    elif reference_price < 50000:
        limit = 7000
    elif reference_price < 70000:
        limit = 10000
    elif reference_price < 100000:
        limit = 15000
    elif reference_price < 150000:
        limit = 30000
    elif reference_price < 200000:
        limit = 40000
    elif reference_price < 300000:
        limit = 50000
    elif reference_price < 500000:
        limit = 70000
    elif reference_price < 700000:
        limit = 100000
    elif reference_price < 1000000:
        limit = 150000
    elif reference_price < 1500000:
        limit = 300000
    elif reference_price < 2000000:
        limit = 400000
    elif reference_price < 3000000:
        limit = 500000
    elif reference_price < 5000000:
        limit = 700000
    elif reference_price < 7000000:
        limit = 1000000
    elif reference_price < 10000000:
        limit = 1500000
    elif reference_price < 15000000:
        limit = 3000000
    elif reference_price < 20000000:
        limit = 4000000
    elif reference_price < 30000000:
        limit = 5000000
    elif reference_price < 50000000:
        limit = 7000000
    else:
        limit = 10000000 # 5000万以上は一律これ以上とする（稀）

    stop_low = max(1, reference_price - limit) # 1円未満にはならない
    stop_high = reference_price + limit
    
    return stop_low, stop_high
