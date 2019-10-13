def compare(self, score1, score2, theshold=0.01, greater_is_better=True):
    sign = 1 if greater_is_better else -1
    sig2 = 1 if score2 > 0 else -1
    if sign * score1 >= sign * (1 - sign * sig2 * theshold) * score2:
        return True
    else:
        return False

def _select(self, slices, group, score, theshold=0.01, fliters=False, greater_is_better=True):
    score_group = [[score[i] for i in slicei_gruop] for slicei_gruop in group]
    select = [np.argmax(i) for i in score_group]  # 选择的在族中的位置
    # for n, best, score_groupi, groupi in zip(range(len(select_gs)), select_gs, score_group, cal_group):
    #     for i, _, index in zip(range(len(groupi)), score_groupi, groupi):
    #         if len(slices_i[groupi[best]]) > len(slices_i[index]) and self.compare(score_groupi[i], score_groupi[best],
    #                                                                            theshold=theshold,
    #                                                                            greater_is_better=greater_is_better):
    #             best = i
    #     select_gs[n] = best

    slices_select = [i[_] for _, i in zip(select, group)]  # 选择的在初始的位置
    slices_select = list(set(slices_select)) if fliters else slices_select
    score_select = [score[_] for _ in slices_select]  # 选择的分数
    selected = [slices[_] for _ in slices_select]  # 选择
    return score_select, selected, slices_select


def select(self, slices, group, estimator_i=0, theshold=0.01, greater_is_better=True):
    score = self.score_all(slices, n_jobs=3, estimator_i=estimator_i)
    return self._select(slices, group, score, theshold=theshold, fliters=False, greater_is_better=greater_is_better)
