#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Base class for Batch GEMM.
"""

# pylint: disable=C0103,W0223

from aitemplate.compiler.base import Tensor

from .gemm_common import gemm


class bmm(gemm):
    """Base class for bmm."""

    def _get_batch_size(self, a: Tensor, b: Tensor):
        self._sanity_check(a, b)

        a_shapes = a._attrs["shape"]
        b_shapes = b._attrs["shape"]
        if len(a_shapes) == 2:
            return b_shapes[0]
        elif len(b_shapes) == 2:
            return a_shapes[0]

        batch_size_a = a_shapes[0]
        batch_size_b = b_shapes[0]
        if batch_size_a != batch_size_b and batch_size_a != 1 and batch_size_b != 1:
            raise RuntimeError(
                "bmm operand A and B should have same batch_size, or batch_size = 1! "
                "Current shape A: {} shape B: {} .".format(a_shapes, b_shapes)
            )

        return a_shapes[0] if a_shapes[0] != 1 else b_shapes[0]

    def _sanity_check(self, a: Tensor, b: Tensor):
        a_shapes = a._attrs["shape"]
        if len(a_shapes) != 2 and len(a_shapes) != 3:
            raise RuntimeError(
                "bmm operand A should have 2 or 3 dimensions! Current shape: {}.".format(
                    a_shapes
                )
            )
        b_shapes = b._attrs["shape"]
        if len(b_shapes) != 2 and len(b_shapes) != 3:
            raise RuntimeError(
                "bmm operand B should have 2 or 3 dimensions! Current shape: {}.".format(
                    b_shapes
                )
            )
        if len(a_shapes) == 2 and len(b_shapes) == 2:
            raise RuntimeError(
                "bmm operand A and B both have 2 dimensions! Use gemm instead."
            )
