(ns nlp-in-action.chapter5
  (:require [org.apache.clojure-mxnet.ndarray :as nd]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.io :as io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.initializer :as initializer]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.visualization :as viz]))


(defn xor-model
  []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1" {:data data :num-hidden 10})
    (sym/activation "tanh" {:data data :act-type "tanh"})
    (sym/fully-connected "fc2" {:data data :num-hidden 1})
    (sym/logistic-regression-output "linear_regression_output" {:data data})))

(def x-train (nd/array [0 0 0 1 1 0 1 1] [4 2] {:ctx (context/gpu)}))
(def y-train (nd/array [0 1 1 0] [4] {:ctx (context/gpu)}) )

(def train-iter 
  (io/ndarray-iter [x-train]
                   {:label-name "linear_regression_output_label"
                    :label [y-train] }))


(defn train-model!
  [model]
  (-> model
      (m/bind {:data-shapes (io/provide-data train-iter)
               :label-shapes (io/provide-label train-iter)})
      (m/init-params {:initializer (initializer/xavier) })
      (m/init-optimizer {:optimizer (optimizer/sgd {:learning-rate 0.1})})
      (m/fit {:train-data train-iter :num-epoch 100})))

#_ (train-model!  (m/module (xor-model) {:contexts [(context/gpu)]}))



(defn render-model!
  "Render the `model-sym` and saves it as a pdf file in `path/model-name.pdf`"
  [{:keys [model-name model-sym input-data-shape path]}]
  (let [dot (viz/plot-network
             model-sym
             {"data" input-data-shape}
             {:title model-name
              :node-attrs {:shape "oval" :fixedsize "false"}})]
    (viz/render dot  model-name path)))

(def model-render-dir "model_render")

#_ (render-model! {:model-name "xormodel" :model-sym (xor-model) :input-data-shape [1 2] :path model-render-dir})
