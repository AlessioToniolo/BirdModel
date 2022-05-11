import * as tf from "@tensorflow/tfjs-node";

/*
const datasetConfig = {
    columnConfigs: {
        classindex: {
            isLabel: true
        }
    }
}
*/

const csvUrl = "file://./assets/birds.csv";
const dataset = tf.data.csv(csvUrl);

export { dataset };
