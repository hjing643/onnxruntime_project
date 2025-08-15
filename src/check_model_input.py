import onnx

def read_onnx_model(onnx_path):
    print("read model from ", onnx_path)
    model = onnx.load(onnx_path)
    for opset in model.opset_import:
        print(opset.domain, opset.version)

    # 3. 列出所有输入
    print("\nInputs:")
    for inp in model.graph.input:
        # 如果是 initializer（权重）也会被当成输入，下面过滤只保留真正的网络输入
        if inp.name not in {init.name for init in model.graph.initializer}:
            shape = [d.dim_value if (d.dim_value > 0) else "?" 
                    for d in inp.type.tensor_type.shape.dim]
            print(f"  • {inp.name}: shape={shape}")

    # 4. 列出所有输出
    print("\nOutputs:")
    for out in model.graph.output:
        shape = [d.dim_value if (d.dim_value > 0) else "?" 
                for d in out.type.tensor_type.shape.dim]
        print(f"  • {out.name}: shape={shape}")

    # 5. 统计算子类型（大致了解模型组成）
    from collections import Counter
    ops = [node.op_type for node in model.graph.node]
    cnt = Counter(ops)
    print("\nOperator counts:")
    for op, num in cnt.most_common():
        print(f"  {op}: {num}") 

if __name__ == "__main__":
    read_onnx_model("./filedepends/models/mnist-8.onnx")
    print("finished")
