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
 * GPT-2 preprocessing layers.
 */

/* Original source: keras-nlp/models/gpt2/gpt2_tokenizer.py */
import { Tensor, serialization } from '@tensorflow/tfjs-core';

import { LayerArgs } from '../../../../engine/topology';
// import { NotImplementedError, ValueError } from '../../../../errors';
import { Preprocessor } from '../preprocessor';
import { GPT2Tokenizer } from './gpt2_tokenizer';

export declare interface GPT2PreprocessorArgs extends LayerArgs {
  /**
   * A GPT2Tokenizer instance.
   */
  tokenizer: GPT2Tokenizer;

  /**
   * The length of the packed inputs.
   * Defaults to 1024.
   */
  sequenceLength?: number;

  /**
   * If `true`, the preprocessor will prepend the tokenizer start token to each
   * input sequence.
   * Defaults to `true`.
   */
  addStartToken?: boolean;

  /**
   * If `true`, the preprocessor will prepend the tokenizer end token to each
   * input sequence.
   * Defaults to `true`.
   */
  addEndToken?: boolean;
}

export declare interface GPT2PreprocessorOptions {
  /**
   * A string, `tf.Tensor`, or list of strings.
   */
  x: string|Tensor|string[];

  /**
   * Any label data. Will be passed through unaltered.
   */
  y?: any;

  /**
   * Any label weight data. Will be passed through unaltered.
   */
  sampleWeight?: any;

  /**
   * Pass to override the configured `sequenceLength` of the layer.
   */
  sequenceLength?: number;
}

/**
 * GPT2 preprocessing layer which tokenizes and packs inputs.
 *
 * This preprocessing layer will do 2 things:
 *
 * - Tokenize the inputs using the `tokenizer`.
 * - Construct a dictionary with keys `"tokenIds"`, `"paddingMask"`, that can
 *     be passed directly to a `GPT2Backbone`.
 *
 * The call method of this layer accepts three arguments, `x`, `y`, and
 * `sampleWeight`. `x` can be a string or tensor representing a single
 * segment, a list of strings representing a batch of single segments,
 * or a list of tensors representing multiple segments to be packed together.
 * `y` and `sampleWeight` are both optional, can have any format, and will be
 * passed through unaltered.
 *
 * `GPT2Preprocessor` forces the input to have only one segment, as GPT2 is
 * mainly used for generation tasks. For tasks having multi-segment inputs
 * like "glue/mnli", please use a model designed for classification purposes
 * such as BERT or RoBERTa.
 *
 * Examples:
 *
 * Directly calling the layer on data.
 * ```js
 * const features =  ['a quick fox.', 'a fox quick.'];
 * const vocabulary =
 *    new Map([['<|endoftext|>', 0], ['a', 4], ['Ġquick', 5], ['Ġfox', 6]]);
 * const merges =
 *    ['Ġ q', 'u i', 'c k', 'ui ck', 'Ġq uick', 'Ġ f', 'o x', 'Ġf ox'];
 * const tokenizer = GPT2Tokenizer({vocabulary, merges});
 *
 * const preprocessor = GPT2Preprocessor({tokenizer});
 * preprocessor.call(tensor(['the quick brown fox jumped.']))[0].print();
 * ```
 */
export class GPT2Preprocessor extends Preprocessor {
  // private readonly tokenizer: GPT2Tokenizer;
  // private readonly sequenceLength: number;
  // private readonly addStartToken: boolean;
  // private readonly addEndToken: boolean;

  constructor(args: GPT2PreprocessorArgs) {
    console.log('');
    super(args);
  }

  override getConfig(): serialization.ConfigDict {
    const config = super.getConfig();
    // In the constructor, we pass the list of special tokens to the
    // `unsplittableTokens` arg of the superclass' constructor. Hence, we
    // delete it from the config here.
    delete config.unsplittableTokens;
    return config;
  }
}
serialization.registerClass(GPT2Preprocessor);
