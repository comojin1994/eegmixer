import torch
import torchvision
import torch.nn.functional as F


class Compose(torchvision.transforms.Compose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, x, y):
        return torch.from_numpy(x).type(torch.FloatTensor), y


class RandomChannelPermutation:
    def __init__(self):
        pass

    def __call__(self, x):
        return x[:, torch.randperm(x.shape[1]), :]


class RandomChannelMasking:
    def __init__(self, min_num_channels: int = 3, mean_value: float = 0.5):
        self.min_num_channels = min_num_channels
        self.mean_value = mean_value

        # TODO : 랜덤 selection 분포 변경, label도 같이 변경

    def __call__(self, x, y):
        num_channels = x.shape[1]

        # 최소 개수 이상 랜덤하게 선택 (비복원추출)
        count = torch.randint(self.min_num_channels, num_channels + 1, (1,)).item()
        selected_channels = torch.randperm(num_channels)[:count]  # 비복원추출 적용
        selected_channels = selected_channels.sort().values

        # 마스킹 적용
        x[:, selected_channels, :] = self.mean_value

        # 선택되지 않은 채널 찾기
        all_indices = torch.arange(num_channels)
        unselected_channels = all_indices[~torch.isin(all_indices, selected_channels)]

        # 새로운 순서로 정렬
        new_order = torch.cat((unselected_channels, selected_channels))
        x = x[:, new_order, :]

        selected_ratio = count / num_channels
        not_y = torch.ones_like(y) - y
        y *= selected_ratio
        not_y *= (1 - selected_ratio) / (y.shape[0] - 1)
        y += not_y

        return x, y


class Padding:
    def __init__(self, padding_size: int = 1125, padding_value: float = 0):
        self.padding_size = padding_size
        self.padding_value = padding_value

    def __call__(self, x, y):

        x = torch.cat(
            (
                x,
                torch.ones((x.shape[0], x.shape[1], self.padding_size - x.shape[2]))
                * self.padding_value,
            ),
            dim=2,
        )

        return x, y


class MinMaxNormalization:
    def __init__(self):
        pass

    def __call__(self, x):
        x = (x - x.min()) / (x.max() - x.min())

        return x


class BaselineCorrection:
    def __init__(self, baseline_ptr: int):
        self.baseline_ptr = baseline_ptr

    def __call__(self, x):
        baseline_value = x[:, : self.baseline_ptr].mean(-1, keepdims=True)
        x = x - baseline_value
        return x[:, self.baseline_ptr :]


class Interpolate:
    def __init__(self, target_size: int = 22):
        self.target_size = target_size

    def __call__(self, x, y):
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.target_size, x.shape[2]),
            mode="bilinear",
        )
        return x.squeeze(0), y


class ZScoreNormalization:
    mean_dict = {
        "bcic2a": 0.4907,
        "bcic2b": 0.4833,
        "zhou": 0.4560,
    }
    std_dict = {
        "bcic2a": 0.0371,
        "bcic2b": 0.0420,
        "zhou": 0.1098,
    }

    def __init__(self, dataset_name: str):
        self.mean = self.mean_dict[dataset_name]
        self.std = self.std_dict[dataset_name]

    def __call__(self, x, y):
        x = (x - x.mean()) / x.std()
        return x, y


class FastDistanceAwareSelector:
    def __init__(self, num_channels=3, max_retries=50):
        self.N = num_channels
        self.R = max_retries

    def __call__(self, x, y):
        M = x.shape[1]
        self.d = M // self.N

        # 1) 거리 행렬 사전 계산 (CPU/GPU 상관없이 한 번)
        coords = torch.arange(M, device=x.device).unsqueeze(1)
        D = torch.abs(coords - coords.T)  # [M,M]

        best_sel = None
        for _ in range(self.R):
            selected_mask = torch.zeros(M, dtype=torch.bool, device=x.device)
            min_dist = torch.full((M,), float("inf"), device=x.device)

            for _ in range(self.N):
                # soft 제약 기반 확률
                weights = torch.where(
                    min_dist >= self.d,
                    torch.ones_like(min_dist),
                    1.0 / (min_dist + 1e-6),
                )
                # 이미 선택된 채널은 확률 0
                weights[selected_mask] = 0
                if weights.sum() == 0:
                    break
                probs = weights / weights.sum()
                idx = torch.multinomial(probs, 1).item()
                selected_mask[idx] = True
                min_dist = torch.minimum(min_dist, D[idx])

            # 성공 시 바로 종료
            if selected_mask.sum().item() == self.N:
                best_sel = selected_mask
                break

        # Fallback: 부족할 때 가까운 채널 채우기
        if best_sel is None or best_sel.sum().item() < self.N:
            sel = (
                best_sel
                if best_sel is not None
                else torch.zeros(M, dtype=torch.bool, device=x.device)
            )
            rem = (~sel).nonzero(as_tuple=False).squeeze()
            need = self.N - sel.sum().item()
            extra = rem[torch.topk(min_dist[rem], need).indices]
            sel[extra] = True
            best_sel = sel

        chosen = torch.nonzero(best_sel, as_tuple=False).squeeze().tolist()
        x = x[:, sorted(chosen), :]
        return x, y
