from concurrent.futures import ThreadPoolExecutor, Future
import concurrent.futures
import contextlib
from types import TracebackType
from typing import Any, Callable, Iterable, Iterator, ParamSpec, TypeVar, overload

import concurrent

_T = TypeVar("_T")    # noqa: WVarDec01
_P = ParamSpec("_P")    # noqa: WVarDec01

class ThreadPoolOrCurrentThreadExecutor(contextlib.AbstractContextManager):
    def __init__(self, worker_count: int | None, pre_submit_rate_for_lazy: float = 1) -> None:
        if worker_count is None:
            self.__pool: ThreadPoolExecutor | None = None
            self.__lazy_submit: int = 1
        else:
            self.__pool = ThreadPoolExecutor(worker_count)
            self.__lazy_submit = int(worker_count * (pre_submit_rate_for_lazy + 1))

    @property
    def worker_count(self):
        if self.__pool:
            return self.__pool._max_workers
        return None

    def submit(self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]:
        if self.__pool:
            return self.__pool.submit(fn, *args, **kwargs)
        
        future: Future[_T] = Future()
        try:
            result: _T = fn(*args, **kwargs)
            future.set_result(result)

            e: BaseException
        except BaseException as e:
            future.set_exception(e)
        return future

    def map(self, 
            fn: Callable[..., _T], 
            *iterables: Iterable[Any]) -> Iterator[_T]:
        if self.__pool:
            return self.__pool.map(fn, *iterables)
        return map(fn, *iterables)
    
    @overload
    def lazy_submit[_T](self, fn: Callable[[], _T], /) -> Iterator[Future[_T]]: ...
    @overload
    def lazy_submit[T1, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1], _T], 
        arg1: Iterable[T1], /) -> Iterator[Future[_T]]: ...
    @overload
    def lazy_submit[T1, T2, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1, T2], _T], 
        arg1: Iterable[T1], 
        arg2: Iterable[T2], /) -> Iterator[Future[_T]]: ...
    @overload
    def lazy_submit[T1, T2, T3, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1, T2, T3], _T], 
        arg1: Iterable[T1], 
        arg2: Iterable[T2],
        arg3: Iterable[T3], /) -> Iterator[Future[_T]]: ...
    @overload
    def lazy_submit[T1, T2, T3, T4, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1, T2, T3, T4], _T], 
        arg1: Iterable[T1], 
        arg2: Iterable[T2],
        arg3: Iterable[T3], 
        arg4: Iterable[T4], /) -> Iterator[Future[_T]]: ...
    @overload
    def lazy_submit[T1, T2, T3, T4, T5, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1, T2, T3, T4, T5], _T], 
        arg1: Iterable[T1], 
        arg2: Iterable[T2],
        arg3: Iterable[T3], 
        arg4: Iterable[T4], 
        arg5: Iterable[T5], /) -> Iterator[Future[_T]]: ...
    @overload
    def lazy_submit[T1, T2, T3, T4, T5, T6, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1, T2, T3, T4, T5, T6], _T], 
        arg1: Iterable[T1], 
        arg2: Iterable[T2],
        arg3: Iterable[T3], 
        arg4: Iterable[T4], 
        arg5: Iterable[T5],
        arg6: Iterable[T6], /) -> Iterator[Future[_T]]: ...
    @overload
    def lazy_submit[T1, T2, T3, T4, T5, T6, T7, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1, T2, T3, T4, T5, T6, T7], _T], 
        arg1: Iterable[T1], 
        arg2: Iterable[T2],
        arg3: Iterable[T3], 
        arg4: Iterable[T4], 
        arg5: Iterable[T5],
        arg6: Iterable[T6],
        arg7: Iterable[T7], /) -> Iterator[Future[_T]]: ...
    @overload
    def lazy_submit[T1, T2, T3, T4, T5, T6, T7, T8, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], _T], 
        arg1: Iterable[T1], 
        arg2: Iterable[T2],
        arg3: Iterable[T3], 
        arg4: Iterable[T4], 
        arg5: Iterable[T5],
        arg6: Iterable[T6],
        arg7: Iterable[T7],
        arg8: Iterable[T8], /) -> Iterator[Future[_T]]: ...
    def lazy_submit(    # noqa: WVarDec02
            self,
            fn: Callable[..., _T], 
            *args: Iterable[Any]) -> Iterator[Future[_T]]:
        args_zip: zip[tuple] = zip(*args)
        futures: set[Future[_T]] = set()
        
        for _ in range(self.__lazy_submit):
            try:
                arg: tuple = next(args_zip)
            except StopIteration:
                break
            futures.add(self.submit(fn, *arg))

        while futures:
            done: set[Future[_T]]
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            
            future: Future
            for future in done:
                futures.remove(future)
                yield future
                try:
                    arg = next(args_zip)
                except StopIteration:
                    continue
                futures.add(self.submit(fn, *arg))
    
    @overload
    def lazy_submit_result[_T](self, fn: Callable[[], _T], /) -> Iterator[_T]: ...
    @overload
    def lazy_submit_result[T1, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1], _T], 
        arg1: Iterable[T1], /) -> Iterator[_T]: ...
    @overload
    def lazy_submit_result[T1, T2, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1, T2], _T], 
        arg1: Iterable[T1], 
        arg2: Iterable[T2], /) -> Iterator[_T]: ...
    @overload
    def lazy_submit_result[T1, T2, T3, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1, T2, T3], _T], 
        arg1: Iterable[T1], 
        arg2: Iterable[T2],
        arg3: Iterable[T3], /) -> Iterator[_T]: ...
    @overload
    def lazy_submit_result[T1, T2, T3, T4, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1, T2, T3, T4], _T], 
        arg1: Iterable[T1], 
        arg2: Iterable[T2],
        arg3: Iterable[T3], 
        arg4: Iterable[T4], /) -> Iterator[_T]: ...
    @overload
    def lazy_submit_result[T1, T2, T3, T4, T5, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1, T2, T3, T4, T5], _T], 
        arg1: Iterable[T1], 
        arg2: Iterable[T2],
        arg3: Iterable[T3], 
        arg4: Iterable[T4], 
        arg5: Iterable[T5], /) -> Iterator[_T]: ...
    @overload
    def lazy_submit_result[T1, T2, T3, T4, T5, T6, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1, T2, T3, T4, T5, T6], _T], 
        arg1: Iterable[T1], 
        arg2: Iterable[T2],
        arg3: Iterable[T3], 
        arg4: Iterable[T4], 
        arg5: Iterable[T5],
        arg6: Iterable[T6], /) -> Iterator[_T]: ...
    @overload
    def lazy_submit_result[T1, T2, T3, T4, T5, T6, T7, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1, T2, T3, T4, T5, T6, T7], _T], 
        arg1: Iterable[T1], 
        arg2: Iterable[T2],
        arg3: Iterable[T3], 
        arg4: Iterable[T4], 
        arg5: Iterable[T5],
        arg6: Iterable[T6],
        arg7: Iterable[T7], /) -> Iterator[_T]: ...
    @overload
    def lazy_submit_result[T1, T2, T3, T4, T5, T6, T7, T8, _T](    # noqa: WVarDec02
        self, 
        fn: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], _T], 
        arg1: Iterable[T1], 
        arg2: Iterable[T2],
        arg3: Iterable[T3], 
        arg4: Iterable[T4], 
        arg5: Iterable[T5],
        arg6: Iterable[T6],
        arg7: Iterable[T7],
        arg8: Iterable[T8], /) -> Iterator[_T]: ...
    def lazy_submit_result(    # noqa: WVarDec02
            self,
            fn: Callable[..., _T], 
            *args: Iterable[Any]) -> Iterator[_T]:
        future: Future
        for future in self.lazy_submit(fn, *args):
            yield future.result()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        if self.__pool:
            self.__pool.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self):
        return self

    def __exit__(self,
                 exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 traceback: TracebackType | None):
        if self.__pool:
            self.__pool.__exit__(exc_type, exc_value, traceback)
