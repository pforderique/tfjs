/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 *  Base class for Backbone models.
 */

/* Original source: keras_nlp/models/gpt2/gpt2_backbone.py */
import { serialization } from '@tensorflow/tfjs-core';

import { ContainerArgs } from '../../../engine/container';
import { LayersModel } from '../../../engine/training';
import { RandomNormal } from '../../../initializers';
import { input } from 'tfjs-layers/src/exports';
import { Embedding } from '../../embeddings';
import { SymbolicTensor } from 'tfjs-layers/src/engine/topology';
import { PositionEmbedding } from '../modeling/position_embedding';
import { add } from 'tfjs-layers/src/exports_layers';
import { Dropout } from '../../core';
import { TransformerDecoder } from '../modeling/transformer_decoder';
import { getActivation } from 'tfjs-layers/src/activations';
import { LayerNormalization } from '../../normalization';

function gpt2KernelInitializer(stddev = 0.02) {
  return new RandomNormal({stddev});
}

export interface GPT2BackboneArgs extends ContainerArgs {
  /**
   * Integer. The size of the token vocabulary.
   */
  vocabularySize: number;

  /**
   * Integer. The number of transformer layers.
   */
  numLayers: number;

  /**
   * Integer. The number of attention heads for each transformer.
   * The hidden size must be divisible by the number of attention heads.
   */
  numHeads: number;

  /**
   * Integer. The size of the transformer encoding and pooler layers.
   */
  hiddenDim: number;

  /**
   * Integer. The output dimension of the first Dense layer in a two-layer
   * feedforward network for each transformer.
   */
  intermediateDim: number;

  /**
   * Float. Dropout probability for the Transformer encoder.
   * Defaults to 0.2.
   */
  dropout?: number;

  /**
   * Integer. The maximum sequence length that this encoder can consume.
   * If `null`, `maxSequenceLength` uses the value from sequence length.
   * This determines the variable shape for positional embeddings.
   * Defaults to 1024.
   */
  maxSequenceLength?: number;
}

/**
 * GPT-2 core network with hyperparameters.
 *
 * This network implements a Transformer-based decoder network,
 * Generative Pretrained Transformer-2 (GPT-2), as described in
 * ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
 * It includes the embedding lookups and transformer layers.
 *
 * The default constructor gives a fully customizable, randomly initialized
 * GPT-2 model with any number of layers, heads, and embedding
 * dimensions. To load preset architectures and weights, use the `fromPreset`
 * constructor.
 *
 * Disclaimer: Pre-trained models are provided on an "as is" basis, without
 * warranties or conditions of any kind. The underlying model is provided by a
 * third party and subject to a separate license, available
 * [here](https://github.com/openai/gpt-2).
 *
 *
 * Example usage:
 * ```js
 * const tokenIds = tf.ones([1, 12]), dtype="int32");
 * const paddingMask = tf.tensor(
 *  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], 'int32');
 *
 * # Pretrained GPT-2 decoder.
 * model = GPT2Backbone.fromPreset("gpt2_base_en");
 * model.apply(inputData, {paddingMask});
 *
 * # Randomly initialized GPT-2 decoder with custom config.
 * model = kerasNlp.models.GPT2Backbone({
 *     vocabularySize: 50257,
 *     numLayers: 12,
 *     numHeads: 12,
 *     hiddenDim: 768,
 *     intermediateDim: 3072,
 *     maxSequenceLength: 1024,
 * });
 * model.apply(inputData, {paddingMask});
 * ```
 */
export class GPT2Backbone extends LayersModel {
  private vocabularySize: number;
  private numLayers: number;
  private numHeads: number;
  private hiddenDim: number;
  private intermediateDim: number;
  private dropout: number;
  private maxSequenceLength: number;

  constructor(args: GPT2BackboneArgs) {
    super(args);
    this.vocabularySize = args.vocabularySize;
    this.numLayers = args.numLayers;
    this.numHeads = args.numHeads;
    this.hiddenDim = args.hiddenDim;
    this.intermediateDim = args.intermediateDim;
    this.dropout = args.dropout ?? 0.1;
    this.maxSequenceLength = args.maxSequenceLength ?? 1024;

    // Inputs
    const tokenIds = input({shape: [null], dtype: 'int32', name: 'token_ids'});
    const paddingMask =
      input({shape: [null], dtype: 'int32', name: 'padding_mask'});

    // Embed tokens, positions.
    const tokenEmbedding = new Embedding({
      inputDim: this.vocabularySize,
      outputDim: this.hiddenDim,
      embeddingsInitializer: gpt2KernelInitializer(0.01),
      name: 'token_embedding',
    }).apply(tokenIds) as SymbolicTensor;

    const positionEmbedding = new PositionEmbedding({
      initializer: gpt2KernelInitializer(0.02),
      sequenceLength: this.maxSequenceLength,
      name: 'position_embedding',
    }).apply(tokenEmbedding) as SymbolicTensor;

    // Sum and apply dropout to embeddings.
    let x = add({name: 'embeddings_add'})
      .apply([tokenEmbedding, positionEmbedding]) as SymbolicTensor;
    x = new Dropout({rate: this.dropout, name: 'embeddings_dropout'})
      .apply(x) as SymbolicTensor;

    // Apply successive transformer decoder blocks.
    for(let i = 0; i < this.numLayers; i++) {
      x = new TransformerDecoder({
        intermediateDim: this.intermediateDim,
        numHeads: this.numHeads,
        dropout: this.dropout,
        layerNormEpsilon: 1e-05,
        // TODO(pforderique): Implement gelu.
        activation: getActivation('relu'),
        kernelInitializer: gpt2KernelInitializer(0.02),
        normalizeFirst: true,
        name: `transformer_layer_${i}`,
      }).apply(x, {decoderPaddingMask: paddingMask}) as SymbolicTensor;
    }

    const sequenceOutput = new LayerNormalization({
      name: 'layer_norm',
      axis: -1,
      epsilon: 1e-05,
      dtype: 'float32',
    }).apply(x) as SymbolicTensor;

    // Instantiate using Functional API Model constructor.
    super({
      inputs: [tokenIds, paddingMask],
      outputs: sequenceOutput,
    });
  }

  override getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      vocabularySize: this.vocabularySize,
      numLayers: this.numLayers,
      numHeads: this.numHeads,
      hiddenDim: this.hiddenDim,
      intermediateDim: this.intermediateDim,
      dropout: this.dropout,
      maxSequenceLength: this.maxSequenceLength,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  get tokenEmbedding() {
    return this.getLayer('token_embedding');
  }

  static presets(cls: serialization.SerializableConstructor<GPT2Backbone>) {
    return {
      gpt2_base_en: {
        metadata: {
          description: '12-layer GPT-2 model where case is maintained.' +
            'Trained on WebText.',
          params: 124439808,
          official_name: 'GPT-2',
          path: 'gpt2',
          modelCard: 'https://github.com/openai/gpt-2/blob/master/model_card.md',
        },
        config: {
          vocabularySize: 50257,
          numLayers: 12,
          numHeads: 12,
          hiddenDim: 768,
          intermediateDim: 3072,
          dropout: 0.1,
          maxSequenceLength: 1024,
        },
        preprocessorConfig: {},
        weightsUrl: 'https://storage.googleapis.com/keras-nlp/models/gpt2_base_en/v1/model.h5',
        weightsHash: 'f4ea6e1b214516dd7de452461ee6e16e',
        vocabularyUrl: 'https://storage.googleapis.com/keras-nlp/models/gpt2_base_en/v1/vocab.json',
        vocabularyHash: 'dffec25a898b1f5e569bec4dffd7e5c0',
        mergesUrl: 'https://storage.googleapis.com/keras-nlp/models/gpt2_base_en/v1/merges.txt',
        mergesHash: '75a37753dd7a28a2c5df80c28bf06e4e',
      },
      // TODO(pforderique): Add more presets and discuss model loading.
    }
  }

}
serialization.registerClass(GPT2Backbone);
