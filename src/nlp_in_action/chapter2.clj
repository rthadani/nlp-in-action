(ns nlp-in-action.chapter2
  (:require [clojure.string :as str] 
            [org.apache.clojure-mxnet.ndarray :as nd]))

(defn bag-of-words
  [sentences-as-words]
  (map frequencies sentences-as-words))

(def sentences
   ["Thomas Jefferson began building Monticello at the age of 26."
    "Construction was done mostly by local masons and carpenters."
    "He moved into the South Pavilion in 1770."
    "Turning Monticello into a neoclassical masterpiece was Jefferson's obsession."]))

(->> sentences
  (map (comp bag-of-words str/split)))
