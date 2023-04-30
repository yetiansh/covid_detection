# !/bin/bash

# Copyright Ye Tian (yetiansh@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

metrics="macro_roc_auc,macro_pr_auc,micro_roc_auc,micro_pr_auc,accuracy"

for backbone in resnet18 resnet50 ; do
    echo "Start testing for $backbone"
    python3 covid_detection/test.py --model=$backbone --num_workers=1 \
        --metrics=$metrics 2>&1 | tee test_$backbone.log
done