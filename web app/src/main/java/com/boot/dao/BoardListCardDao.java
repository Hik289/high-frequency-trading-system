package com.boot.dao;

import com.boot.entity.BoardListCard;

public interface BoardListCardDao {

	void save(BoardListCard listCard);

	BoardListCard getDetailById(Integer cardId);

	void update(BoardListCard listCard);

}
