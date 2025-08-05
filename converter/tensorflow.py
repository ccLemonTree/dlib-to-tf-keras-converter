import os
import shutil

#import tensorflow as tf
#from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

def convert_to_tf_saved_model(keras_model, output_path):
    sess = K.get_session()
    output_node_names = [node.op.name for node in keras_model.outputs]
        
    inputs={'input_image': keras_model.input}
    outputs={t.name: t for t in keras_model.outputs}
    
    exported_model_path = os.path.join(output_path, 'exported')
    
    if os.path.exists(exported_model_path):
        shutil.rmtree(exported_model_path)
    
    tf.saved_model.simple_save(sess,
            os.path.join(output_path, 'exported'),
            inputs=inputs,
            outputs=outputs)
    
def convert_to_tf_frozen_model(keras_model, output_path):        
    import tensorflow as tf
    sess = tf.compat.v1.keras.backend.get_session()
    
    # 获取所有输出节点名称（使用已定义的列表推导式）
    output_node_names = [node.op.name for node in keras_model.outputs]
    # 确保输出节点名称有效
    if not output_node_names:
        raise ValueError("模型没有找到输出节点，请检查模型结构")
    
    output_model_dir = output_path
    output_model_name = 'dlib_face_recognition_resnet_model_v1.pb'

    # 使用收集到的输出节点名称（修复原代码中未使用output_node_names的问题）
    constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess, 
        sess.graph_def, 
        output_node_names  # 直接使用提前收集的节点名列表
    )


    # 保存冻结模型
    tf.io.write_graph(constant_graph, output_model_dir, output_model_name, as_text=False)
    print(f"冻结模型已保存至: {output_model_dir}/{output_model_name}")
