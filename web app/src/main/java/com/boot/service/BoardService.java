package com.boot.service;

import java.util.List;

import com.boot.entity.Board;
import com.boot.entity.BoardList;
import com.boot.entity.BoardListCard;

public interface BoardService {

	void addBoardSave(Board board);
	
	Board getDetailByid(Integer boardId);

	List<Board> getListByTeam(Integer teamId);

	void listSave(BoardList list);

	List<BoardList> getBoardListByBoard(Integer boardId);
	
	BoardList getBoardListDetailById(Integer listId);

	List<BoardListCard> getBoardListCard(Integer boardListId);

	
}
