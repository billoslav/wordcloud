"""
GPU acceleration utilities for wordcloud generation.

This module provides optional GPU acceleration for computationally intensive
operations like collision detection and placement calculations.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, List, Any
import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)

# Try to import GPU libraries
try:
    import cupy as cp # type: ignore
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.debug("CuPy not available, CUDA GPU acceleration disabled")

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    logger.debug("PyOpenCL not available, OpenCL GPU acceleration disabled")


class GPUAccelerator:
    """
    GPU acceleration wrapper for wordcloud operations.
    """
    def __init__(
        self,
        backend: str = 'auto',
        enabled: bool = False,
        validate_results: bool = True,
        validation_samples: int = 128,
    ):
        """
        Initialize GPU accelerator.
        
        Args:
            backend: GPU backend ('cuda', 'opencl', 'auto')
        """
        self.backend = backend
        self.device = None
        self.context = None
        self.queue = None
        self.available = False
        self.enabled = enabled
        self.validate_results = validate_results
        self.validation_samples = validation_samples
        
        if not enabled:
            logger.info("GPU acceleration disabled (opt-in only)")
            return
        
        if backend == 'auto':
            if CUPY_AVAILABLE:
                backend = 'cuda'
            elif OPENCL_AVAILABLE:
                backend = 'opencl'
            else:
                logger.warning("No GPU backend available")
                return
        
        if backend == 'cuda' and CUPY_AVAILABLE:
            try:
                # Check if CUDA is available
                if cp.cuda.is_available():
                    self.device = cp.cuda.Device(0)
                    self.available = True
                    logger.info("CUDA GPU acceleration enabled")
                else:
                    logger.warning("CUDA not available on this system")
            except Exception as e:
                logger.warning(f"Failed to initialize CUDA: {e}")
        
        elif backend == 'opencl' and OPENCL_AVAILABLE:
            try:
                platforms = cl.get_platforms()
                if platforms:
                    devices = platforms[0].get_devices(cl.device_type.GPU)
                    if devices:
                        self.device = devices[0]
                        self.context = cl.Context([self.device])
                        self.queue = cl.CommandQueue(self.context)
                        self.available = True
                        logger.info("OpenCL GPU acceleration enabled")
                    else:
                        logger.warning("No OpenCL GPU devices found")
                else:
                    logger.warning("No OpenCL platforms found")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenCL: {e}")
    
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.available
    
    def accelerate_collision_detection(
        self,
        integral_image: np.ndarray,
        word_width: int,
        word_height: int
    ) -> Optional[np.ndarray]:
        """
        Accelerate collision detection using GPU.
        
        Args:
            integral_image: Integral image array
            word_width: Width of word bounding box
            word_height: Height of word bounding box
            
        Returns:
            Array of valid positions or None if not available
        """
        if not self.available:
            return None
        
        try:
            if self.backend == 'cuda' and CUPY_AVAILABLE:
                # Transfer to GPU
                gpu_image = cp.asarray(integral_image)
                
                # Calculate valid positions using GPU operations
                # This is a simplified example - full implementation would
                # check all possible positions efficiently
                h, w = gpu_image.shape
                valid_positions = []
                
                # Check positions in parallel (simplified)
                for y in range(h - word_height + 1):
                    for x in range(w - word_width + 1):
                        # Calculate area sum using integral image
                        area = (gpu_image[y + word_height, x + word_width] -
                               gpu_image[y, x + word_width] -
                               gpu_image[y + word_height, x] +
                               gpu_image[y, x])
                        
                        if area == 0:  # No collision
                            valid_positions.append((x, y))
                
                gpu_positions = np.array(valid_positions) if valid_positions else None
                if self.validate_results and gpu_positions is not None:
                    if not self._validate_positions(integral_image, gpu_positions, word_width, word_height):
                        logger.warning("GPU collision results failed validation, falling back to CPU")
                        return None
                return gpu_positions
            
            elif self.backend == 'opencl' and OPENCL_AVAILABLE:
                # OpenCL implementation would go here
                logger.debug("OpenCL collision detection not yet implemented")
                return None
        
        except Exception as e:
            logger.warning(f"GPU collision detection failed: {e}, falling back to CPU")
            return None
    
    def accelerate_placement_calculation(
        self,
        positions: List[Tuple[int, int]],
        word_sizes: List[Tuple[int, int]],
        canvas_width: int,
        canvas_height: int
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Accelerate placement calculations using GPU.
        
        Args:
            positions: List of candidate positions
            word_sizes: List of (width, height) tuples
            canvas_width: Canvas width
            canvas_height: Canvas height
            
        Returns:
            List of optimized positions or None if not available
        """
        if not self.available or not positions:
            return None
        
        try:
            if self.backend == 'cuda' and CUPY_AVAILABLE:
                # Transfer data to GPU
                pos_array = cp.asarray(positions)
                sizes_array = cp.asarray(word_sizes)
                
                # Perform calculations on GPU
                # This is a placeholder - full implementation would optimize
                # position selection based on various criteria
                
                # For now, just return original positions
                if self.validate_results:
                    if not self._validate_positions_list(positions, word_sizes, canvas_width, canvas_height):
                        logger.warning("GPU placement results failed validation, falling back to CPU")
                        return None
                return positions
            
            elif self.backend == 'opencl' and OPENCL_AVAILABLE:
                # OpenCL implementation would go here
                logger.debug("OpenCL placement calculation not yet implemented")
                return None
        
        except Exception as e:
            logger.warning(f"GPU placement calculation failed: {e}, falling back to CPU")
            return None

    def _validate_positions(
        self,
        integral_image: np.ndarray,
        positions: np.ndarray,
        word_width: int,
        word_height: int,
    ) -> bool:
        if positions.size == 0:
            return True
        max_samples = min(self.validation_samples, len(positions))
        sample_indices = np.linspace(0, len(positions) - 1, num=max_samples, dtype=int)
        for idx in sample_indices:
            x, y = positions[idx]
            area = (
                integral_image[y + word_height, x + word_width]
                - integral_image[y, x + word_width]
                - integral_image[y + word_height, x]
                + integral_image[y, x]
            )
            if area != 0:
                return False
        return True

    def _validate_positions_list(
        self,
        positions: List[Tuple[int, int]],
        word_sizes: List[Tuple[int, int]],
        canvas_width: int,
        canvas_height: int,
    ) -> bool:
        if len(positions) != len(word_sizes):
            return False
        max_samples = min(self.validation_samples, len(positions))
        sample_indices = np.linspace(0, len(positions) - 1, num=max_samples, dtype=int)
        for idx in sample_indices:
            x, y = positions[idx]
            width, height = word_sizes[idx]
            if x < 0 or y < 0 or x + width > canvas_width or y + height > canvas_height:
                return False
        return True


def create_gpu_accelerator(
    backend: str = 'auto',
    enabled: bool = False,
    validate_results: bool = True,
    validation_samples: int = 128,
) -> Optional[GPUAccelerator]:
    """
    Create a GPU accelerator instance.
    
    Args:
        backend: GPU backend ('cuda', 'opencl', 'auto')
        
    Returns:
        GPUAccelerator instance or None if not available
    """
    accelerator = GPUAccelerator(
        backend=backend,
        enabled=enabled,
        validate_results=validate_results,
        validation_samples=validation_samples,
    )
    return accelerator if accelerator.is_available() else None

