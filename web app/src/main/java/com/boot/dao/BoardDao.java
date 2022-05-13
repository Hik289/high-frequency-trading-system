package com.boot.dao;

import java.util.List;

import org.apache.ibatis.annotations.Param;

import com.boot.entity.Board;
import com.boot.entity.BoardList;
import com.boot.entity.BoardListCard;

public interface BoardDao {

	void addBoardSave(Board board);

	List<Board> getListByTeam(@Param("teamId")Integer teamId);

	void listSave(BoardList list);
	
	/**
	 * board list
	 */
	List<BoardList> getBoardListByBoard(@Param("boardId")Integer boardId);

	List<BoardListCard> getBoardListCard(@Param("boardListId")Integer boardListId);

	Board getDetailByid(@Param("boardId")Integer boardId);

	BoardList getBoardListDetailById(@Param("listId")Integer listId);

}
