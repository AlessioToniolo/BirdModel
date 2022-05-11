import * as tf from "@tensorflow/tfjs-node";
import { xyDataset } from "./src/io.js";

// Create model
const model = tf.sequential();

// Add CNN Layers
/*
model.add(
  tf.layers.conv2d({
    inputShape: [224, 224, 3],
    kernelSize: 3,
    filters: 16,
    strides: 1,
    padding: "same",
    activation: "relu",
  })
);

model.add(tf.layers.maxPooling2d({ poolsize: 2, strides: 2 }));

model.add(
  tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    strides: 1,
    padding: "same",
    activation: "relu",
  })
);

model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

model.add(tf.layers.flatten());
*/

model.add(
  tf.layers.dense({
    units: 1,
    inputShape: [200704],
  })
);

// Train model
async function train() {
  /*
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  */

  model.compile({ optimizer: "sgd", loss: "meanSquaredError" });

  let results = await model.fitDataset(xyDataset, {
    epochs: 1,
    //validationData: validationXYDataset,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch}: ${logs.loss}`);
      },
    },
  });
}

// Train the model
await train();
