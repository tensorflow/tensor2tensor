/**
 * @license
 * Copyright 2018 The Tensor2Tensor Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @fileoverview A set of shared types that will be replaced by js proto types.
 */

/**
 * A typedef for a nlp.nmt.mt_debug_fe.LanguageConfiguration message.
 * This can't be converted to javascript yet because it transitively depends on
 * tensorflow protos that can't be converted to javascript.
 * TODO(kstevens): Remove this typedef when we remove the dependency on
 * non-convertible tensorflow protos.
 * @typedef {{
 *   code: string,
 *   name: string,
 *   hidden: ?boolean,
 * }}
 */
let Language;

/**
 * A typedef for a nlp.nmt.mt_debug_fe.SerializedConfiguration message.
 * This can't be converted to javascript yet because it transitively depends on
 * tensorflow protos that can't be converted to javascript.
 * TODO(kstevens): Remove this typedef when we remove the dependency on
 * non-convertible tensorflow protos.
 * @typedef {{
 *   id: string,
 *   target: string,
 *   source_language: Language,
 *   target_language: Language,
 * }}
 */
let Model;

/**
 * @typedef {{
 *  name: string,
 *    localProbability: number,
 *    cumalitiveProbability: number,
 *    attention: Array<number>,
 *    children: Array<TreeNode>,
 * }}
 */
let TreeNode;

/**
 * @typedef {{
 *   source_tokens: Array<string>,
 *   target_tokens: Array<string>,
 *   weights: !Array<number>
 * }}
 */
let AttentionData;

/**
 * @typedef {{
 *   label: string,
 *   label_id: number,
 *   log_probability: number,
 *   total_log_probability: number,
 *   score: number,
 *   parent_id: number,
 * }}
 */
let Candidate;

/**
 * @typedef {{
 *   id: number,
 *   stepIndex: number,
 *   candidate: !Candidate,
 *   children: !Array<InteractiveNode>,
 * }}
 */
let InteractiveNode;

/**
 * @typedef {{
 *   step_name: string,
 *   segment: !Array<!{
 *     text: string,
 *   }>
 * }}
 */
let QueryProcessingRewriteStep;

/**
 * @typedef {{
 *   source_processing: !Array<!QueryProcessingRewriteStep>,
 *   target_processing: !Array<!QueryProcessingRewriteStep>,
 * }}
 */
let QueryProcessingVisualization;

/**
 * @typedef {{
 *   in_edge_index: !Array<number>,
 *   out_edge_index: !Array<number>,
 * }}
 */
let BeamSearchNode;

/**
 * @typedef {{
 *   label_id: number,
 *   label: string,
 *   log_probability: number,
 *   total_log_probability: number,
 *   score: number,
 *   completed: boolean,
 * }}
 */
let BeamSearchCandidate;

/**
 * @typedef {{
 *   source_index: number,
 *   target_index: number,
 *   data: !BeamSearchCandidate,
 * }}
 */
let BeamSearchEdge;

/**
/**
 * @typedef {{
 *   node: !Array<!BeamSearchNode>,
 *   edge: !Array<!BeamSearchEdge>,
 * }}
 */
let SearchGraphVisualization;

/**
 * @typedef {{
 *   candidate_list: !Array<{
 *     candidate: !Array<!BeamSearchCandidate>,
 *   }>,
 * }}
 */
let GenerateCandidateResponse;

/**
 * @typedef {{
 *   session_id: number,
 * }}
 */
let StartTranslationResponse;
